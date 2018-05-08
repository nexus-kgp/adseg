
# coding: utf-8

# In[1]:

import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from loader.dataloader_SBD import SFBD
from loss import dummyBCE
from models.segmentor import fcn32s
from models.discriminator import StanfordBNet


# def cross_entropy2d(input, target, weight=None, size_average=True):
#     n, c, h, w = input.size()
#     log_p = F.log_softmax(input)
#     log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
#     # log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
#     log_p = log_p[target.view(n, h, w, 8) >= 0]
#     log_p = log_p.view(-1, c)

#     mask = target >= 0
#     target = target[mask]
#     loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
#     if size_average:
#         loss /= mask.data.sum()
#     return loss


def initialize_fcn32s(n_classes):

    segmentor = fcn32s

    try:
        segmentor = segmentor(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=False)   ## change it to True
        segmentor.init_vgg16_params(vgg16)
    except:
        print('Error occured in initialising fcn32s')
        sys.exit(1)

    return segmentor


segmentor = initialize_fcn32s(8)
disc = StanfordBNet()

use_gpu = torch.cuda.is_available()
# use_gpu = False

print(use_gpu)
if use_gpu:
    segmentor = segmentor.cuda()       # using GPU for processing
    disc = disc.cuda()

dataset = SFBD(image_path="./iccv09Data/images/",region_path="./h5py/")

batch_size = 1
train_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=True)



if use_gpu:
    zeros = Variable(torch.zeros((batch_size)).cuda(), requires_grad=False)
    ones = Variable(torch.ones((batch_size)).cuda(), requires_grad=False)
else:
    zeros = Variable(torch.zeros((batch_size)), requires_grad=False)
    ones = Variable(torch.ones((batch_size)), requires_grad=False)

d_loss = nn.BCELoss()

g_optim = optim.Adam(segmentor.parameters(), lr=1e-4)
d_optim = optim.Adam(disc.parameters(), lr=1e-4)

# g = None

count = 0
for i in train_loader:
    if use_gpu:
        sample_image = Variable(i['image'].float().cuda())
        label = Variable(i['region'].float().cuda())
    else:
        sample_image = Variable(i['image'].float())
        label = Variable(i['region'].float())
    # print(sample_image)
    # break
    
    fake_out = segmentor(sample_image)

    disc.zero_grad()
    segmentor.zero_grad()

    d_fake_out = disc(fake_out,sample_image)

    fake_err = d_loss(d_fake_out, zeros)
    fake_err.backward(retain_graph=True)

    d_real_out = disc(label, sample_image)
    real_err = d_loss(d_real_out, ones)
    real_err.backward()

    d_optim.step()

    
    g_err = dummyBCE(fake_out, label) + 0.5*(d_loss(d_fake_out,ones))
    g_err.backward()

    g_optim.step()
    count += 1
    print("done")
    if count == 20:
        break