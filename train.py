
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
from loss import cross_entropy2d
from models.segmentor import fcn32s
from models.discriminator import StanfordBNet
from LRN.local_response_norm import local_response_norm

from PIL import Image

def si(x):
    Image.fromarray(x).show()

def initialize_fcn32s(n_classes):

    segmentor = fcn32s

    try:
        segmentor = segmentor(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=True)   ## change it to True
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

dataset = SFBD(image_path="./traindata/images/",region_path="./traindata/h5py/")

batch_size = 1
train_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=True)



if use_gpu:
    zeros = Variable(torch.zeros((batch_size)).cuda(), requires_grad=False)
    ones = Variable(torch.ones((batch_size)).cuda(), requires_grad=False)
else:
    zeros = Variable(torch.zeros((batch_size)), requires_grad=False)
    ones = Variable(torch.ones((batch_size)), requires_grad=False)

d_loss = nn.BCELoss(size_average=False)

g_optim = optim.Adam(segmentor.parameters(), lr=1e-5)
d_optim = optim.Adam(disc.parameters(), lr=1e-5)

# g = None

fake_loss_d = []
real_loss_d = []
real_loss_gen = []



def train(n_epoch=1,n_co=10):
    for i in range(n_epoch):
        count = 0
        for i in train_loader:
            if use_gpu:
                sample_image = Variable(i['image'].float().cuda())
                sample_image = local_response_norm(sample_image, 3)

                label = Variable(i['region'].float().cuda())
            else:
                sample_image = Variable(i['image'].float())
                sample_image = local_response_norm(sample_image, 3)

                label = Variable(i['region'].float())
            
            fake_out = segmentor(sample_image)

            disc.zero_grad()
            segmentor.zero_grad()

            d_fake_out = disc(fake_out,sample_image)

            fake_err = d_loss(d_fake_out, zeros)
            fake_err.backward(retain_graph=True)
            fake_loss_d.append(fake_err[0].clone().cpu().data.numpy()[0])

            d_real_out = disc(label, sample_image)
            real_err = d_loss(d_real_out, ones)
            real_err.backward()
            real_loss_d.append(real_err[0].clone().cpu().data.numpy()[0])

            d_optim.step()

            
            g_err = cross_entropy2d(fake_out, label) + 0.65*(d_loss(d_fake_out,ones))
            g_err.backward()
            real_loss_gen.append(g_err[0].clone().cpu().data.numpy()[0])

            g_optim.step()
            count += 1
            print("done")
            if count == n_co:
                break


train(n_epoch=12, n_co=len(dataset))