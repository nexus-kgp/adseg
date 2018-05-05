
# coding: utf-8

# In[1]:

import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data

from loader.dataloader_SBD import SFBD
# from loss import cross_entropy2d
from models.segmentor import fcn32s
from models.discriminator import LargeFOV


# In[2]:

# In[2]:

def initialize_fcn32s(n_classes):

    segmentor = fcn32s

    try:
        segmentor = segmentor(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=False)
        segmentor.init_vgg16_params(vgg16)
    except:
        print('Error occured in initialising fcn32s')
        sys.exit(1)

    return segmentor


# In[3]:

segmentor = initialize_fcn32s(8)


# In[4]:

dataset = SFBD(image_path="./iccv09Data/images/",region_path="./h5py/")


# In[5]:

train_loader = data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)


# In[7]:

g = None
for i in train_loader:
    g = Variable(i['image'].float())
    print(segmentor(g))
    break


# In[16]:




# In[ ]:



