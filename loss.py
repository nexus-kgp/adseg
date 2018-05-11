# Author:- Dibya Prakash Das

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

lossfunc = nn.BCELoss(size_average=False)

def cross_entropy2d(input, target):
    _ , c , _ , _ = input.size()
    target.requires_grad = False
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    return lossfunc(input, target)

