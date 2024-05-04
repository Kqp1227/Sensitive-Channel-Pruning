from torch.utils.data import Dataset
from data_augmentation_1 import *
import torch
import os 
import io
import skimage
import pickle
from skimage import io
from torchvision import datasets, transforms
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from sklearn.utils import shuffle


class dataset_handler:
    def __init__(self, args):
        self.args = args
        self.num_class = {
                'isic2019' : 8,
                'fitzpatrick17k' : 114
            }
    def get_num_class(self):
        return self.num_class[self.args.dataset]

    def get_dataset_class(self):
        if self.args.dataset == 'isic2019':
            return ISIC2019
        if self.args.dataset == 'fitzpatrick17k':
            return fitzpatrick17k

    def get_dataset(self, is_one_batch=False):
        if is_one_batch:
            batch_size = 1
        else:
            batch_size = self.args.batch_size
            
        dataset = self.get_dataset_class()(batch_size=batch_size, args=self.args)

        return dataset
    
def get_weighted_sampler(df, label_level = 'low'):
    class_sample_count = np.array(df[label_level].value_counts().sort_index())
    class_weight = 1. / class_sample_count
    samples_weight = np.array([class_weight[t] for t in df[label_level]])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
    return sampler

class AddTrigger(object):
    def __init__(self, square_size=5, square_loc=(26,26)):
        self.square_size = square_size
        self.square_loc = square_loc

    def __call__(self, pil_data):
        square = Image.new('L', (self.square_size, self.square_size), 255)
        pil_data.paste(square, self.square_loc)
        return pil_data

class ISIC2019_dataset_transform(Dataset):

    def __init__(self, df=None, root_dir=None, transform=True, feature_dict=None):
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.feature_dict = feature_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.df.loc[self.df.index[idx], 'image']+'.jpg')
        image = io.imread(img_name)
        # some images have alpha channel, we just not ignore alpha channel
        if (image.shape[0] > 3):
            image = image[:,:,:3]
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)
        if self.transform:
            image = self.transform(image)
        label = self.df.loc[self.df.index[idx], 'low']
        gender = self.df.loc[self.df.index[idx], 'gender']
        feature = {}
    
        return image, label, gender , idx

def ISIC2019_holdout_gender(df, holdout_set: str = 'none'):
    if holdout_set == "0":
        remain_df = df[df.gender==1].reset_index(drop=True)
    elif holdout_set == "1":
        remain_df = df[df.gender==0].reset_index(drop=True)
    else:
        remain_df = df
    return remain_df

class ISIC2019:
    def __init__(self, args, batch_size=64, add_trigger=False, model_name=None, feature_dict=None, input_size=128):
        self.batch_size = batch_size
        self.num_classes = 8
        self.image_size = input_size

        predefined_root_dir = '/afs/crc.nd.edu/user/q/qkong2/Private/ISIC_2019_Training_Input' # specify the image dir
        train_df = pd.read_csv('./isic2019_split/isic2019_train_pretraining.csv')
        vali_df = pd.read_csv('./isic2019_split/isic2019_val_pretraining.csv')
        test_df = pd.read_csv('./isic2019_split/isic2019_test_pretraining.csv')
        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
        sampler = get_weighted_sampler(train_df, label_level='low')
        train_transform = ISIC2019_Augmentations(is_training=True, image_size=self.image_size, input_size=self.image_size).transforms
        test_transform = ISIC2019_Augmentations(is_training=False, image_size=self.image_size, input_size=self.image_size).transforms
        aug_trainset =  ISIC2019_dataset_transform(df=train_df, root_dir=predefined_root_dir, transform=train_transform, feature_dict=feature_dict)
        self.aug_train_loader = torch.utils.data.DataLoader(aug_trainset, drop_last=True, batch_size=self.batch_size, sampler=sampler, **kwargs)
        train_dataset = ISIC2019_dataset_transform(df=train_df, root_dir=predefined_root_dir,transform=train_transform, feature_dict=feature_dict)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, drop_last=True, batch_size=self.batch_size, sampler=sampler, **kwargs)
        vali_dataset= ISIC2019_dataset_transform(df=vali_df, root_dir=predefined_root_dir, transform=test_transform, feature_dict=feature_dict)
        self.vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test_dataset= ISIC2019_dataset_transform(df=test_df, root_dir=predefined_root_dir, transform=test_transform, feature_dict=feature_dict)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


        if add_trigger:  #skip this
            self.trigger_test_set = None
            self.trigger_test_loader = None

class fitzpatrick17k:
    def __init__(self, args, batch_size=64, add_trigger=False, model_name=None, input_size=224):
        self.batch_size = batch_size
        self.num_classes = 114
        
        augmentation_rand = transforms.Compose(
            [transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
            transforms.ToTensor()]
            # transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192))]
            )

        augmentation_sim = transforms.Compose(
            [transforms.RandomResizedCrop(224,scale=(0.8,1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])

        predefined_root_dir = '/afs/crc.nd.edu/user/q/qkong2/fairprune/finalfitz17k/finalfitz17k' # specify the image dir
        train_df = pd.read_csv('/afs/crc.nd.edu/user/q/qkong2/Private/Fair-Multi-Exit-Framework-master/fitzpatrick17k/fitzpatrick17k_train.csv')
        vali_df = pd.read_csv('/afs/crc.nd.edu/user/q/qkong2/Private/Fair-Multi-Exit-Framework-master/fitzpatrick17k/fitzpatrick17k_vali.csv')
        test_df = pd.read_csv('/afs/crc.nd.edu/user/q/qkong2/Private/Fair-Multi-Exit-Framework-master/fitzpatrick17k/fitzpatrick17k_test.csv')
        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
        sampler = get_weighted_sampler(train_df, label_level='low')
        
        train_transform = [augmentation_rand, augmentation_sim]

        self.image_size = input_size
        self.crop_size = input_size
        print("input size:", input_size)
        train_transform = ISIC2019_Augmentations(is_training=True, image_size=self.image_size, input_size=self.crop_size).transforms
        train_SNNL_transform = ISIC2019_Augmentations(is_training=False, image_size=self.image_size, input_size=self.crop_size).transforms
        test_transform = ISIC2019_Augmentations(is_training=False, image_size=self.image_size, input_size=self.crop_size).transforms
        aug_trainset =  Fitzpatrick17k_dataset_transform(df=train_df, root_dir=predefined_root_dir, transform=train_transform)
        self.aug_train_loader = torch.utils.data.DataLoader(aug_trainset, drop_last=True, batch_size=self.batch_size, sampler=sampler, **kwargs)
        train_dataset = Fitzpatrick17k_dataset_transform(df=train_df, root_dir=predefined_root_dir, transform=train_transform)
        train_SNNL_dataset = Fitzpatrick17k_dataset_transform(df=train_df, root_dir=predefined_root_dir, transform=train_SNNL_transform)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, drop_last=True, batch_size=self.batch_size, sampler=sampler, **kwargs)
        self.train_SNNL_loader = torch.utils.data.DataLoader(train_SNNL_dataset, drop_last=True, batch_size=self.batch_size, shuffle=False, **kwargs)
        vali_dataset= Fitzpatrick17k_dataset_transform(df=vali_df, root_dir=predefined_root_dir, transform=test_transform)
        self.vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        test_dataset= Fitzpatrick17k_dataset_transform(df=test_df, root_dir=predefined_root_dir, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        if add_trigger:  #skip this
            self.trigger_test_set = None
            self.trigger_test_loader = None

class Fitzpatrick17k_dataset_transform(Dataset):

    def __init__(self, df=None, root_dir=None, transform=None):
        """
        Args:
            train: True for training, False for testing
            transform (callable, optional): Optional transform to be applied
                on a sample.
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        assert df is not None
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.df.loc[self.df.index[idx], 'hasher']+'.jpg')
        img = Image.open(img_name)
        image = io.imread(img_name)
        if(len(image.shape) < 3):
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], 'hasher']
        high = self.df.loc[self.df.index[idx], 'high']
        mid = self.df.loc[self.df.index[idx], 'mid']
        low = self.df.loc[self.df.index[idx], 'low']
        fitzpatrick = self.df.loc[self.df.index[idx], 'fitzpatrick']
        if 1 <= fitzpatrick <= 3:
            skin_color_binary = 0
        elif 4 <= fitzpatrick <= 6:
            skin_color_binary = 1
        else:
            skin_color_binary = 1
        if self.transform:
            image = self.transform(image)
#        img1 = self.transform[0](img)
#        img2 = self.transform[1](img)
    
        label = self.df.loc[self.df.index[idx], 'low']
        sample = {
                    'image': image,
                    'high': high,
                    'mid': mid,
                    'low': low,
                    'hasher': hasher,
                    'fitzpatrick': fitzpatrick,
                    'skin_color_binary': skin_color_binary,
                }
        return image, label, skin_color_binary, idx
#        return [img1,img2], label, skin_color_binary, idx
