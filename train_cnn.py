import torch
import time
import data
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import aux_funcs as af

from util.custom_loss import *
from data import *
from tqdm import tqdm
from model import *
from util.bce_acc import *
from sklearn import metrics
import torch.nn.functional as F
from fairness_metric import *
from torchvision import models
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision

# vanilla scores (us to calculate FATE score)
# Fitz
ACC_B = 0.5513
FC_B = 0.1636

# ISIC
ACC_B = 0.7862
FC_B = 0.0209

def compute_fate(acc_m, acc_b, fc_m, fc_b, lamb=1.0):
    fate = ( (acc_m - acc_b) / acc_b ) - ( lamb * (fc_m - fc_b) / fc_b )
    return fate

def cnn_test(model, loader, writer, device='cpu', epoch=0):
    model.eval()
    gender_list = []
    label_list = []
    y_pred_list = []
    groupAcc = []
    with torch.no_grad():
        for batch in tqdm(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            gender = batch[2].to(device)
            output = model(b_x)
            _, pred = torch.max(output, 1)

            label_list.append(b_y.detach().cpu().numpy())
            y_pred_list.append(pred.detach().cpu().numpy())
            gender_list.append(gender.cpu().numpy())
    label_list = np.concatenate(label_list)
    y_pred_list = np.concatenate(y_pred_list)
    gender_list = np.concatenate(gender_list)
    group0_f1_score, group1_f1_score, fairness_metrics = compute_fairness_metrics(
        label_list, y_pred_list, gender_list)
    #print(fairness_metrics)
        
    EOdd_abs = fairness_metrics["fairness/EOdds_abs"]
    acc = fairness_metrics["avg/F1"]
    fate = compute_fate(acc, ACC_B, EOdd_abs, FC_B, 1.0)
    writer.add_scalar("fate", fate, epoch)
    print("fate:", fate)

    for k, v in fairness_metrics.items():
        print('{}:{:.4f}'.format(k, v))
        writer.add_scalar(k, v, epoch)
        
        
            
        


def cnn_train(model, data, epochs, optimizer, scheduler, device='cuda', tensor_board_path='', models_path='', args=None):

    writer = SummaryWriter(tensor_board_path)

    for epoch in range(1, epochs):
        cnn_test(model, data.test_loader, writer, device, epoch)
        CE_loss = []
        label_list = []
        y_pred_list = []
        sensitive_group_list = []
        cur_lr = af.get_lr(optimizer)
        train_loader = data.train_loader
        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        for x, y, sensitive_group, idx in tqdm(train_loader):
            b_x = x.to(device)   # batch x
            b_y = y.to(device)   # batch y
            b_sensitive_group = sensitive_group.to(device)
            output = model(b_x)  # cnn final output

            _, preds = torch.max(output, 1)

            criterion = af.get_loss_criterion('')
            loss = criterion(output, b_y)
            optimizer.zero_grad()           # clear gradients for this training step
            loss.mean().backward()           # backpropagation, compute gradients
            optimizer.step()

            CE_loss.append(loss.mean())
            label_list.append(b_y.detach().cpu().numpy())
            y_pred_list.append(preds.detach().cpu().numpy())
            sensitive_group_list.append(sensitive_group.numpy())

        scheduler.step()
        label_list = np.concatenate(label_list)
        y_pred_list = np.concatenate(y_pred_list)
        sensitive_group_list = np.concatenate(sensitive_group_list)
        end_time = time.time()

        epoch_time = int(end_time-start_time)

        print('CE Loss: {}'.format(sum(CE_loss) / len(CE_loss)))
        print('Epoch took {} seconds.'.format(epoch_time))
        writer.add_scalar('CE Loss: ', sum(CE_loss) / len(CE_loss), epoch)
        writer.add_scalar("Lr/train", cur_lr, epoch)
        # cnn_test(model, data.test_loader, device)
        
        #print(model)
        if epoch % 10 == 0:
            #if epoch > 20:
                #if os.path.exists('{}/{}.pth'.format(models_path, epoch-10)):
                    #os.remove('{}/{}.pth'.format(models_path, epoch-10))
            torch.save(model, '{}/{}.pth'.format(models_path, epoch))
        print("model save to:", models_path)
        print('Start testing...')
        cnn_test(model, data.test_loader, writer, device, epoch)

import torch.nn.utils.prune as prune
class ChannelPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "channel"

    def __init__(self, channels):
        self.channels = channels

    def compute_mask(self, weight):
        mask = torch.ones_like(weight)
        for channel in channels:
            mask[:, channel * 7 * 7: (channel + 1) * 7 * 7] = 0
            # mask[self.channels] = 0
        print("mask:", mask)
        return mask
        
class OutChannelPruning(prune.BasePruningMethod):
    PRUNING_TYPE = "channel"

    def __init__(self, channels):
        self.channels = channels

    def compute_mask(self, weight):
        mask = torch.ones_like(weight)
        for channel in self.channels:
            mask[channel, :, :, :] = 0
        print("mask:", mask)
        return mask
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_cnn')
    parser.add_argument('--training_title', type=str, default='vgg_prune_5',
                        help='')
    parser.add_argument('--epochs', type=int, default=201,
                        help='')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='')
    parser.add_argument('--dataset', type=str, default='fitzpatrick17k',
                        help='')
    parser.add_argument('--model', type=str, default='vgg11',
                        help='')
    parser.add_argument('--class_num', type=int, default=114,
                        help='')
    #parser.add_argument('--threshold', type=float, default=50.0)
    parser.add_argument('--gpu', type=int, default=0)
    
    
    args = parser.parse_args()
    training_title = args.training_title
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()
    models_path = 'networks/{}/{}'.format(af.get_random_seed(), training_title)
    print(models_path)

    
    tensor_board_path = 'runs/{}/train_models{}'.format(
        training_title, af.get_random_seed())
    af.create_path(models_path)
    af.create_path(tensor_board_path)
    af.create_path('outputs/{}'.format(training_title))
    af.set_logger(
        'outputs/{}/train_models{}'.format(training_title, af.get_random_seed()))

    print("Arguments: ")
    argument_list = ""
    for arg in vars(args):
        argument_list += " --{} {}".format(arg, getattr(args, arg))
    print(argument_list)
    
    model = torch.load("200.pth") # input pretrained model here
    print(model)

    # prune the model
    channels = [59, 133, 141, 146, 166, 210, 222] # channels to prune (use SNNL.py to calculate and input here)
    layer = model.features[18] # layer to prune
    
    prune.custom_from_mask(layer, name="weight", mask=OutChannelPruning(channels).compute_mask(layer.weight))
    
    model.to(device)
    
    # finetune the model
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    ds_handler = dataset_handler(args)
    dataset = ds_handler.get_dataset()
    one_batch_dataset = ds_handler.get_dataset(is_one_batch=True)
    cnn_train(model, dataset, args.epochs, optimizer, scheduler,
              device, tensor_board_path, models_path, args=args)
    cnn_test(model, dataset.test_loader, device)
