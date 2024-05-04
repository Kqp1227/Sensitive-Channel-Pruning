# aux_funcs.py
# contains auxiliary functions for optimizers, internal classifiers, confusion metric
# conversion between CNNs and mes and also plotting

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
from model import *
from data import  ISIC2019

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


def get_random_seed():
    return 121 # 121 and  or 120(new epochs)

def set_random_seeds():
    torch.manual_seed(get_random_seed())
    np.random.seed(get_random_seed())
    random.seed(get_random_seed())

def extend_lists(list1, list2, items):
    list1.append(items[0])
    list2.append(items[1])

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
    return CrossEntropyLoss(reduction='none').cuda()
    
    
    

