import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def get_statistics(self, dataloader, batch_size=128, num_workers=2, model=None):
    
        if model is not None:
            model.eval()

        Y_pred_set = []
        Y_set = []
        S_set = []
        total = 0
        for i, data in enumerate(dataloader):
            inputs, _, sen_attrs, targets, indexes = data
            Y_set.append(targets)  # sen_attrs = -1 means no supervision for sensitive group
            S_set.append(sen_attrs)

            if self.cuda:
                inputs = inputs.cuda()
                groups = sen_attrs.cuda()
            if model is not None:
                outputs = model(inputs)
                Y_pred_set.append(torch.argmax(outputs, dim=1))
            total += inputs.shape[0]

        Y_set = torch.cat(Y_set)
        S_set = torch.cat(S_set)
        Y_pred_set = torch.cat(Y_pred_set) if len(Y_pred_set) != 0 else torch.zeros(0)
        return Y_pred_set.long(), Y_set.long().cuda(), S_set.long().cuda()

    # Vectorized version for DP & multi-class
def get_error_and_violations_DP(self, y_pred, label, sen_attrs, num_groups, num_classes):
    acc = torch.mean(y_pred == label)
    total_num = len(y_pred)
    violations = torch.zeros((num_groups, num_classes))

    for g in range(num_groups):
        for c in range(num_classes):
            pivot = len(torch.where(y_pred == c)[0]) / total_num
            group_idxs = torch.where(sen_attrs == g)[0]
            group_pred_idxs = torch.where(torch.logical_and(sen_attrs == g, y_pred == c))[0]
            violations[g, c] = len(group_pred_idxs)/len(group_idxs) - pivot
    return acc, violations

# Vectorized version for EO & multi-class
def get_error_and_violations_EO(self, y_pred, label, sen_attrs, num_groups, num_classes):
    acc = torch.mean((y_pred == label).float())
    violations = torch.zeros((num_groups, num_classes))
    for g in range(num_groups):
        for c in range(num_classes):
            class_idxs = torch.where(label == c)[0]
            pred_class_idxs = torch.where(torch.logical_and(y_pred == c, label == c))[0]
            pivot = len(pred_class_idxs)/len(class_idxs)
            group_class_idxs = torch.where(torch.logical_and(sen_attrs == g, label == c))[0]
            group_pred_class_idxs = torch.where(torch.logical_and(torch.logical_and(sen_attrs == g, y_pred == c), label == c))[0]
            violations[g, c] = len(group_pred_class_idxs)/len(group_class_idxs) - pivot
    print('violations', violations)
    return acc, violations

# update weight
def debias_weights(self, label, sen_attrs, extended_multipliers, num_groups, num_classes):
    weights = torch.zeros(len(label))
    w_matrix = torch.sigmoid(extended_multipliers)  # g by c
    weights = w_matrix[sen_attrs, label]
    if self.slmode and self.version == 2:
        weights[sen_attrs == -1] = 0.5
    return weights