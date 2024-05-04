# aux_funcs.py
# contains auxiliary functions for optimizers, internal classifiers, confusion metric
# conversion between CNNs and SDNs and also plotting

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import os.path
import torch.optim as optim
import sys
import itertools as it

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

from bisect import bisect_right
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
# from model import *
# from data import  ISIC2019

# to log the output of the experiments to a file
class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr

        self.log = open('{}.{}'.format(log_file, mode), "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()

def set_logger(log_file):
    sys.stdout = Logger(log_file, 'out')

# the learning rate scheduler
class MultiStepMultiLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gammas = gammas
        super(MultiStepMultiLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            cur_milestone = bisect_right(self.milestones, self.last_epoch)
            new_lr = base_lr * np.prod(self.gammas[:cur_milestone])
            new_lr = round(new_lr,8)
            lrs.append(new_lr)
        return lrs

# flatten the output of conv layers for fully connected layers
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# the formula for feature reduction in the internal classifiers
def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1
      
# the internal classifier for all SDNs
class InternalClassifier(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(InternalClassifier, self).__init__()
        #red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.output_channels = output_channels

        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels*input_size*input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1))
            self.linear = nn.Linear(output_channels*red_input_size*red_input_size, num_classes)
            self.forward = self.forward_w_pooling

    def forward_w_pooling(self, x):
        avgp = self.alpha*self.max_pool(x)
        maxp = (1 - self.alpha)*self.avg_pool(x)
        mixed = avgp + maxp
        return self.linear(mixed.view(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))


def get_random_seed():
    return 121 # 121 and  or 120(new epochs)

def get_subsets(input_list, sset_size):
    return list(it.combinations(input_list, sset_size))

def set_random_seeds():
    torch.manual_seed(get_random_seed())
    np.random.seed(get_random_seed())
    random.seed(get_random_seed())

def extend_lists(list1, list2, items):
    list1.append(items[0])
    list2.append(items[1])

def get_dataset(dataset, batch_size=128, add_trigger=False):
    if dataset == 'isic2019':
        return load_ISIC2019(batch_size)


def load_ISIC2019(batch_size):
    isic2019 = ISIC2019(batch_size=batch_size)
    return isic2019


def get_output_relative_depths(model):
    total_depth = model.init_depth
    output_depths = []

    for layer in model.layers:
        total_depth += layer.depth

        if layer.no_output == False:
            output_depths.append(total_depth)

    total_depth += model.end_depth

    #output_depths.append(total_depth)
 
    return np.array(output_depths)/total_depth, total_depth


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def model_exists(models_path, model_name):
    return os.path.isdir(models_path+'/'+model_name)

def get_nth_occurance_index(input_list, n):
    if n == -1:
        return len(input_list) - 1
    else:
        return [i for i, n in enumerate(input_list) if n == 1][n]

def get_lr(optimizers):
    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']


def get_pytorch_device():
    device = 'cpu'
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = 'cuda'
    return device


def get_loss_criterion(dataset_name=''):
    if dataset_name == '':
        return CrossEntropyLoss(reduction='none').cuda()
        # Added reduction='none', the output will become [batch_size]. simply use loss.mean() to get the averaged loss.
    if dataset_name == 'celebA':
        return BCEWithLogitsLoss().cuda()
    
def get_sdn(model=''):
    """

    Args:
        model (str, optional): _description_. Defaults to ''.

    Returns:
        nn.module: selected models 
    """
    if model == 'VGG_Early_Exits':
        return VGG_Early_Exits
    if model == 'ResNet_Early_Exits':
        return ResNet_Early_Exits
    if model == 'ResNet18_Early_Exits_CelebA':
        return ResNet18_Early_Exits_CelebA
    if model == 'ResNet18_Early_Exits':
        return ResNet18_Early_Exits
    
    

