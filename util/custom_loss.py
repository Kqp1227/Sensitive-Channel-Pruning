import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1).detach().cuda()
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        return x/x.sum()

class HA_Loss(nn.Module):
    def __init__(self, w_m, bin_width):
        super(HA_Loss, self).__init__()
        self.w_m = w_m
        self.bin_width = bin_width

    def forward(self, f_s, groups, labels):  
        softhist = SoftHistogram(bins=10, min=0, max=100, sigma=6)
        student = F.softmax(f_s.view(f_s.shape[0], -1), dim=1)[: :, 1]*100
        
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        D00 = softhist(student[(labels == 0) * (groups == 0)]) + 1e-5
        D01 = softhist(student[(labels == 0) * (groups == 1)]) + 1e-5
        D10 = softhist(student[(labels == 1) * (groups == 0)]) + 1e-5
        D11 = softhist(student[(labels == 1) * (groups == 1)]) + 1e-5
        loss_0 = kl_loss(D00.log(), D01) +  kl_loss(D01.log(), D00)
        loss_1 = kl_loss(D10.log(), D11) +  kl_loss(D11.log(), D10)        
        loss = (loss_1 + loss_0)/2
        
        return self.w_m * loss

    def gaussian_kernel(self, x, bin_center):
        std = 0.0215
        bin_count = torch.exp(-(x - bin_center)**2/(2*(std**2)))
        return torch.sum(bin_count)
    
    def get_histogram(self, x):
        Dis = []
        for i in range(int(1//self.bin_width)):
            bin_center = (i+1) * self.bin_width
            Dis.append(self.gaussian_kernel(x, bin_center) + 1e-5)

        Dis = torch.tensor(Dis)
        Dis =  Dis/torch.sum(Dis)
        return Dis

class MMDLoss_Multi_class_Distill(nn.Module):
    def __init__(self, w_m, sigma, num_groups, num_classes, kernel):
        super(MMDLoss_Multi_class_Distill, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels, jointfeature=False):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1).detach()

        mmd_loss = 0

        if jointfeature:
            K_TS, sigma_avg = self.pdist(teacher, student,
                              sigma_base=self.sigma, kernel=self.kernel)
            K_TT, _ = self.pdist(teacher, teacher, sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)
            K_SS, _ = self.pdist(student, student,
                              sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        else:
            with torch.no_grad():
                _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

            for c in range(self.num_classes):
                if len(teacher[labels==c]) == 0:
                    continue
                for g in range(self.num_groups):
                    if len(student[(labels==c) * (groups == g)]) == 0:
                        continue
                    K_TS, _ = self.pdist(teacher[labels == c][:,c-1], student[(labels == c) * (groups == g)][:,c-1],
                                                 sigma_base=self.sigma, sigma_avg=sigma_avg,  kernel=self.kernel)
                    K_SS, _ = self.pdist(student[(labels == c) * (groups == g)][:,c-1], student[(labels == c) * (groups == g)][:,c-1],
                                         sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                    K_TT, _ = self.pdist(teacher[labels == c][:,c-1], teacher[labels == c][:,c-1], sigma_base=self.sigma,
                                         sigma_avg=sigma_avg, kernel=self.kernel)

                    mmd_loss_TPR = K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()
                    
                    
                    K_TS, _ = self.pdist(teacher[labels != c][:,c-1], student[(labels != c) * (groups == g)][:,c-1],
                                                 sigma_base=self.sigma, sigma_avg=sigma_avg,  kernel=self.kernel)
                    K_SS, _ = self.pdist(student[(labels != c) * (groups == g)][:,c-1], student[(labels != c) * (groups == g)][:,c-1],
                                         sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                    K_TT, _ = self.pdist(teacher[labels != c][:,c-1], teacher[labels != c][:,c-1], sigma_base=self.sigma,
                                         sigma_avg=sigma_avg, kernel=self.kernel)

                    mmd_loss_FPR = K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()
                    
                    mmd_loss += mmd_loss_TPR + mmd_loss_FPR

        loss = (1/2) * self.w_m * mmd_loss

        return loss

    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if len(e1.shape) < 2 or len(e2.shape) < 2:
                e1 = torch.unsqueeze(e1, 1)
                e2 = torch.unsqueeze(e2, 1)
                
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()
                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg

class MMDLoss(nn.Module):
    def __init__(self, w_m, sigma, num_groups, num_classes, kernel):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels, jointfeature=False):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1).detach()

        mmd_loss = 0

        if jointfeature:
            K_TS, sigma_avg = self.pdist(teacher, student,
                              sigma_base=self.sigma, kernel=self.kernel)
            K_TT, _ = self.pdist(teacher, teacher, sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)
            K_SS, _ = self.pdist(student, student,
                              sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        else:
            with torch.no_grad():
                _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

            for c in range(self.num_classes):
                if len(teacher[labels==c]) == 0:
                    continue
                for g in range(self.num_groups):
                    if len(student[(labels==c) * (groups == g)]) == 0:
                        continue
                    K_TS, _ = self.pdist(teacher[labels == c], student[(labels == c) * (groups == g)],
                                                 sigma_base=self.sigma, sigma_avg=sigma_avg,  kernel=self.kernel)
                    K_SS, _ = self.pdist(student[(labels == c) * (groups == g)], student[(labels == c) * (groups == g)],
                                         sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                    K_TT, _ = self.pdist(teacher[labels == c], teacher[labels == c], sigma_base=self.sigma,
                                         sigma_avg=sigma_avg, kernel=self.kernel)

                    mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        loss = (1/2) * self.w_m * mmd_loss

        return loss

    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
#                prod = e1.t() @ e2
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()
                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg