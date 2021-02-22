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
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, x, target):
        return focal_loss(F.cross_entropy(x, target, reduction='none', weight=self.weight), self.gamma)


def upper_triangle(matrix):
    upper = torch.triu(matrix, diagonal=0)
    #diagonal = torch.mm(matrix, torch.eye(matrix.shape[0]))
    diagonal_mask = torch.eye(matrix.shape[0]).cuda()
    return upper * (1.0 - diagonal_mask)

def regularizer(W, regularizer_hp, num_of_im, num_of_sk):
    number_of_sketches = torch.from_numpy(num_of_im).float().cuda()
    number_of_images = torch.from_numpy(num_of_sk).float().cuda()
    # Regularization
    # print("W shape:", W.shape)
    # W is of shape [mc, hidden_layers]
    mc = W.shape[0]
    w_expand1 = W.unsqueeze(0)
    w_expand2 = W.unsqueeze(1)
    wx = (w_expand2 - w_expand1)**2
    w_norm_mat = torch.sum((w_expand2 - w_expand1)**2, dim=-1)
    w_norm_upper = upper_triangle(w_norm_mat)
    mu = 2.0 / (mc**2 - mc) * torch.sum(w_norm_upper)
    delta = number_of_sketches + number_of_images
    delta = regularizer_hp/delta
    residuals = upper_triangle((w_norm_upper - (mu + delta))**2)
    rw = 2.0 / (mc**2 - mc) * torch.sum(residuals)
    return rw

if __name__ == "__main__":
    torch.manual_seed(0)
    W=torch.rand(128,100,4096)
    #rdx= torch.sum(torch.tensor([regularizer(W[i]) for i in range(W.shape[0])]))
    #print(rdx)
    for i in range(W.shape[0]):
        print(regularizer(W[i],1))
