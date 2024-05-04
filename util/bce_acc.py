from __future__ import print_function, absolute_import
import torch

__all__ = ["accuracy", "accuracy_bce"]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_bce(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size, label_number = target.size()

        pred = (output >= 0.5).view(1, -1)
        correct = pred.eq(target.view(1, -1))

        correct_k = correct.view(-1).float().sum(0)
        top1 = correct_k.mul_(100.0 / batch_size / label_number)

        return top1
    
def val_accuracy(pred, ta, sa, ta_cls, sa_cls, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = ta.size(0)

        pred = pred.t()
        correct = pred.eq(ta.view(1, -1).expand_as(pred))

        group=[]
        group_num=[]
        for i in range(ta_cls):
            sa_group=[]
            sa_group_num=[]
            for j in range(sa_cls):
                eps=1e-8
                sa_group.append(((sa==j)*(ta==i)*(correct==1)).float().sum() *(100 /(((sa==j)*(ta==i)).float().sum()+eps)))
                sa_group_num.append(((sa==j)*(ta==i)).float().sum()+eps)
            group.append(sa_group)
            group_num.append(sa_group_num)
       
        res=(correct==1).float().sum()*(100.0 / batch_size)
        
        return res,group,group_num