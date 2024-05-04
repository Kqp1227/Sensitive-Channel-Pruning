# Refer to: https://github.com/chiuhaohao/Fair-Multi-Exit-Framework

import numpy as np
import torch
import aux_funcs as af
from data import *
from tqdm import tqdm
import sys
import torchvision
import torch.nn as nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class SNNLCrossEntropy():
    STABILITY_EPS = 0.00001

    def __init__(self,
                 temperature=100.,
                 layer_names=None,
                 factor=-10.,
                 optimize_temperature=True,
                 cos_distance=True):

        self.temperature = temperature
        self.factor = factor
        self.optimize_temperature = optimize_temperature
        self.cos_distance = cos_distance

    @staticmethod
    def pairwise_euclid_distance(A, B):
        """Pairwise Euclidean distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise Euclidean between A and B.
        """
        batchA = A.shape[0]
        batchB = B.shape[0]

        sqr_norm_A = torch.reshape(torch.pow(A, 2).sum(axis=1), [1, batchA])
        sqr_norm_B = torch.reshape(torch.pow(B, 2).sum(axis=1), [batchB, 1])
        inner_prod = torch.matmul(B, A.T)

        tile_1 = torch.tile(sqr_norm_A, [batchB, 1])
        tile_2 = torch.tile(sqr_norm_B, [1, batchA])
        return (tile_1 + tile_2 - 2 * inner_prod)

    @staticmethod
    def pairwise_cos_distance(A, B):

        """Pairwise cosine distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise cosine between A and B.
        """
        normalized_A = torch.nn.functional.normalize(A, dim=1)
        normalized_B = torch.nn.functional.normalize(B, dim=1)
        distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
        min_clip_distances = tf.math.maximum(distances, 0.0)
        return min_clip_distances

    @staticmethod
    def fits(A, B, temp, cos_distance):
        if cos_distance:
            distance_matrix = SNNLCrossEntropy.pairwise_cos_distance(A, B)
        else:
            distance_matrix = SNNLCrossEntropy.pairwise_euclid_distance(A, B)

        return torch.exp(-(distance_matrix / temp))

    @staticmethod
    def pick_probability(x, temp, cos_distance):
        """Row normalized exponentiated pairwise distance between all the elements
        of x. Conceptualized as the probability of sampling a neighbor point for
        every element of x, proportional to the distance between the points.
        :param x: a matrix
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or euclidean distance
        :returns: A tensor for the row normalized exponentiated pairwise distance
                  between all the elements of x.
        """
        f = SNNLCrossEntropy.fits(x, x, temp, cos_distance) - torch.eye(x.shape[0]).cuda()
        return f / (SNNLCrossEntropy.STABILITY_EPS + f.sum(axis=1).unsqueeze(1))

    @staticmethod
    def same_label_mask(y, y2):
        """Masking matrix such that element i,j is 1 iff y[i] == y2[i].
        :param y: a list of labels
        :param y2: a list of labels
        :returns: A tensor for the masking matrix.
        """
        return (y == y2.unsqueeze(1)).squeeze().to(torch.float32)

    @staticmethod
    def masked_pick_probability(x, y, temp, cos_distance):
        """The pairwise sampling probabilities for the elements of x for neighbor
        points which share labels.
        :param x: a matrix
        :param y: a list of labels for each element of x
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or Euclidean distance
        :returns: A tensor for the pairwise sampling probabilities.
        """
        return SNNLCrossEntropy.pick_probability(x, temp, cos_distance) * \
               SNNLCrossEntropy.same_label_mask(y, y)

    @staticmethod
    def SNNL(x, y, temp=1, cos_distance=True):
        """Soft Nearest Neighbor Loss
        :param x: a matrix.
        :param y: a list of labels for each element of x.
        :param temp: Temperature.
        :cos_distance: Boolean for using cosine or Euclidean distance.
        :returns: A tensor for the Soft Nearest Neighbor Loss of the points
                  in x with labels y.
        """
        summed_masked_pick_prob = SNNLCrossEntropy.masked_pick_probability(x, y, temp, cos_distance).sum(axis=1)
        return -torch.log(SNNLCrossEntropy.STABILITY_EPS + summed_masked_pick_prob).mean()


class SoftNearestNeighborLoss(nn.Module):
    def __init__(self,
                 temperature=1,
                 cos_distance=True):
        super(SoftNearestNeighborLoss, self).__init__()

        self.temperature = temperature
        self.cos_distance = cos_distance

    def pairwise_cos_distance(self, A, B):
        query_embeddings = torch.nn.functional.normalize(A, dim=1)
        key_embeddings = torch.nn.functional.normalize(B, dim=1)
        distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
        return distances

    def forward(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        eps = 1e-9

        pairwise_dist = self.pairwise_cos_distance(embeddings, embeddings)
        pairwise_dist = pairwise_dist / self.temperature
        negexpd = torch.exp(-pairwise_dist)

        # creating mask to sample same class neighboorhood
        batch_size = labels.size(0)
        pairs_y = labels.repeat(batch_size, 1)
        #         pairs_y = torch.broadcast_to(labels, (batch_size, batch_size))
        mask = pairs_y == torch.transpose(pairs_y, 0, 1)
        mask = mask.float()

        # creating mask to exclude diagonal elements
        ones = torch.ones([batch_size, batch_size], dtype=torch.float32).cuda()
        dmask = ones - torch.eye(batch_size, dtype=torch.float32).cuda()

        # all class neighborhood
        alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)
        # same class neighborhood
        sacn = torch.sum(torch.multiply(negexpd, mask), dim=1)

        # adding eps for numerical stability
        # in case of a class having a single occurance in batch
        # the quantity inside log would have been 0
        loss = -torch.log((sacn + eps) / alcn).mean()
        return loss


import numpy as np
import torch
from torchvision import models

class SNNLCrossEntropy():
    STABILITY_EPS = 0.00001

    def __init__(self,
                 temperature=100.,
                 layer_names=None,
                 factor=-10.,
                 optimize_temperature=True,
                 cos_distance=True):

        self.temperature = temperature
        self.factor = factor
        self.optimize_temperature = optimize_temperature
        self.cos_distance = cos_distance

    @staticmethod
    def pairwise_euclid_distance(A, B):
        """Pairwise Euclidean distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise Euclidean between A and B.
        """
        batchA = A.shape[0]
        batchB = B.shape[0]

        sqr_norm_A = torch.reshape(torch.pow(A, 2).sum(axis=1), [1, batchA])
        sqr_norm_B = torch.reshape(torch.pow(B, 2).sum(axis=1), [batchB, 1])
        inner_prod = torch.matmul(B, A.T)

        tile_1 = torch.tile(sqr_norm_A, [batchB, 1])
        tile_2 = torch.tile(sqr_norm_B, [1, batchA])
        return (tile_1 + tile_2 - 2 * inner_prod)

    @staticmethod
    def pairwise_cos_distance(A, B):

        """Pairwise cosine distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise cosine between A and B.
        """
        normalized_A = torch.nn.functional.normalize(A, dim=1)
        normalized_B = torch.nn.functional.normalize(B, dim=1)
        distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
        min_clip_distances = tf.math.maximum(distances, 0.0)
        return min_clip_distances

    @staticmethod
    def fits(A, B, temp, cos_distance):
        if cos_distance:
            distance_matrix = SNNLCrossEntropy.pairwise_cos_distance(A, B)
        else:
            distance_matrix = SNNLCrossEntropy.pairwise_euclid_distance(A, B)

        return torch.exp(-(distance_matrix / temp))

    @staticmethod
    def pick_probability(x, temp, cos_distance):
        """Row normalized exponentiated pairwise distance between all the elements
        of x. Conceptualized as the probability of sampling a neighbor point for
        every element of x, proportional to the distance between the points.
        :param x: a matrix
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or euclidean distance
        :returns: A tensor for the row normalized exponentiated pairwise distance
                  between all the elements of x.
        """
        f = SNNLCrossEntropy.fits(x, x, temp, cos_distance) - torch.eye(x.shape[0]).cuda()
        return f / (SNNLCrossEntropy.STABILITY_EPS + f.sum(axis=1).unsqueeze(1))

    @staticmethod
    def same_label_mask(y, y2):
        """Masking matrix such that element i,j is 1 iff y[i] == y2[i].
        :param y: a list of labels
        :param y2: a list of labels
        :returns: A tensor for the masking matrix.
        """
        return (y == y2.unsqueeze(1)).squeeze().to(torch.float32)

    @staticmethod
    def masked_pick_probability(x, y, temp, cos_distance):
        """The pairwise sampling probabilities for the elements of x for neighbor
        points which share labels.
        :param x: a matrix
        :param y: a list of labels for each element of x
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or Euclidean distance
        :returns: A tensor for the pairwise sampling probabilities.
        """
        return SNNLCrossEntropy.pick_probability(x, temp, cos_distance) * \
               SNNLCrossEntropy.same_label_mask(y, y)

    @staticmethod
    def SNNL(x, y, temp=1, cos_distance=True):
        """Soft Nearest Neighbor Loss
        :param x: a matrix.
        :param y: a list of labels for each element of x.
        :param temp: Temperature.
        :cos_distance: Boolean for using cosine or Euclidean distance.
        :returns: A tensor for the Soft Nearest Neighbor Loss of the points
                  in x with labels y.
        """
        summed_masked_pick_prob = SNNLCrossEntropy.masked_pick_probability(x, y, temp, cos_distance).sum(axis=1)
        return -torch.log(SNNLCrossEntropy.STABILITY_EPS + summed_masked_pick_prob).mean()

class modified_model(nn.Module):
    def __init__(self, cnn_model, class_num=114):
        super(modified_model, self).__init__()
        self.f = nn.ModuleList()
        self.g = None
        for name, module in cnn_model.named_children():
            if isinstance(module, nn.Linear):
                self.g = module
                continue
            self.f.append(module)

    def forward(self, x):
        for layer in self.f:
            x = layer(x)
        final_out = self.g(torch.flatten(x, start_dim=1))
        return final_out

def dump_feature(x, model):
    output = []
    output.append(x)
    for layer in model.features:
        print("layer:", layer)
        if isinstance(layer, nn.Sequential):
            output.append(torch.flatten(x, start_dim=1))
            # print("layer:", layer)
        x = layer(x)
        feature_map = x
    final_feature = torch.flatten(x, start_dim=1)
    output.append(final_feature)
    return output, feature_map

def extract_feature_vgg(x, model):
    #model = model.module
    #features_model = nn.Sequential(*list(model.features.children())[0:22])
    avgpool_model = model.avgpool
    for layer in model.features[:19]:
        x = layer(x)
    #print(x.shape)
    features = x
    #features = avgpool_model(x)
    #print(features.shape)
    return features

def extract_feature(x, model):
    #    model = model.module
    sequential_model = nn.Sequential()
    
    i = 0
    for module_list in model.f:
        module = module_list  # Access the first module within the ModuleList
        sequential_model.add_module(f"module_{len(sequential_model)}", module[0])
        i = i + 1
        if i > 21:
            break

    features_model = sequential_model
    features = features_model(x)
    # print(features.shape)

    return features
    

class CustomVGG(nn.Module):
    def __init__(self, vgg_model):
        super(CustomVGG, self).__init__()
        self.vgg_model = vgg_model

        self.features_map_1 = nn.Sequential(*list(vgg_model.features.children())[:14])
        self.features_map_2 = nn.Sequential(*list(vgg_model.features.children())[13:17])
        self.features_map_3 = nn.Sequential(*list(vgg_model.features.children())[16:19])
        self.after_features_map = nn.Sequential(*list(vgg_model.features.children())[18:])
        self.avgpool = self.vgg_model.avgpool
        self.classifier = self.vgg_model.classifier
        #self.fc = nn.Linear(64 * 7 * 7, 10)  

    def forward(self, x):
        for layer in self.features_map_1:
            x = layer(x)
        output_feature_map_1 = x
        for layer in self.features_map_2:
            x = layer(x)
        output_feature_map_2 = x
        for layer in self.features_map_3:
            x = layer(x)
        output_feature_map_3 = x
        
        
        x = self.after_features_map(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        
        full_output = x
        return output_feature_map_1, output_feature_map_2, output_feature_map_3, full_output

# get (n) sensitive channels with low SNNL scores
def get_min_n(data, n): 
    data_tmp = data.copy()
    data_tmp.sort()
    min_n = data_tmp[n]
    return min_n

def main():
    device = "cuda"
    vgg11 = torch.load("200.pth") # input pretrained model here
    vgg11 = vgg11.to(device)
    layer = vgg11.classifier[0]

    layer_conv = vgg11.features[18]
    batch_size = 128
    dataset = fitzpatrick17k(batch_size=batch_size, args=None)
    loss = SoftNearestNeighborLoss()
    loss_sen = [[], [], [], [], [], []]
    loss_label = [[], [], [], [], [], []]
    feature_list = [[], [], [], [], [], []]

    import csv
    all_feature_map_SNNL = {}
    batch_id = 0
    csv_file_path = 'SNNL_dic_tmp.csv'
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:

        for image, label, skin_color_binary, idx in tqdm(dataset.train_SNNL_loader):
            image = image.cuda()
            #     print(image)
            label = label.cuda()  # 'low'
            sensitive = skin_color_binary.cuda()  # skin_scale
            #     print(image.shape)
            #     features, feature_map = dump_feature(image, vgg11)
            # feature_map = extract_feature(image, vgg11)
            feature_map = extract_feature_vgg(image, vgg11)
            print(feature_map.shape)
            #     print(feature_map.shape)
            feature_map = feature_map.squeeze(0)
            #     print(feature_map.shape)
        
            split_features = torch.chunk(feature_map, feature_map.size(1), dim=1)
            #     print(len(split_features))
            #     print(split_features)
            feature_map_SNNL = []
            for idx in range(len(split_features)):
                split_feature = split_features[idx].view(split_features[idx].shape[0], -1)
                loss_SNNL = loss(split_feature, sensitive).detach().cpu().numpy()
                feature_map_SNNL.append(loss_SNNL.item())
            all_feature_map_SNNL[batch_id] = feature_map_SNNL
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(feature_map_SNNL)
            batch_id += 1
    print(all_feature_map_SNNL)

    array_list = np.array(list(all_feature_map_SNNL.values()))
    averaged_array = np.mean(array_list, axis=0)
    averaged_list = averaged_array.tolist()
        
    data_average = np.array(averaged_list)
    x = np.arange(0, data_average.shape[0])

    torch_tensor = torch.from_numpy(data_average)

    min_5 = get_min_n(data_average, 5) # get sensitive channels with low SNNL score (n = 5, 10, 15, 20, ...)
    print("min_5:", min_5)
    mask = data_average < get_min_n(data_average, 5)
    print("min_5_channels:", x[mask])

if __name__ == '__main__':
    main()
    
    
    