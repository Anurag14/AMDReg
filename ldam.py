import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


def cb_focal_loss(class_num_list):
    beta = 0.99
    effective_num = 1.0 - np.power(beta, class_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    return FocalLoss(weight=per_cls_weights, gamma=0.5).cuda()

def focal_loss_weighted(input_values, gamma, path_weights):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = path_weights * (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss_weighted(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss_weighted, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target, path_weights):
        if type(path_weights) is list:
            path_weights = torch.FloatTensor(path_weights).cuda()
        return focal_loss_weighted(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma, path_weights)

def cb_focal_loss_weighted(class_num_list):
    beta = 0.99
    effective_num = 1.0 - np.power(beta, class_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(class_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    return FocalLoss_weighted(weight=per_cls_weights, gamma=0.5).cuda()
