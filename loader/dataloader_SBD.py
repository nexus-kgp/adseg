from torchvision import transforms,datasets
from torch.utils.data.dataset import Dataset
import torch

import PIL
import numpy as np
import h5py

import subprocess


# In[2]:

class SFBD(Dataset):
    
    """
        A custom dataloader for Stanford Background Dataset.
        http://dags.stanford.edu/projects/scenedataset.html
        
        It works after some preprocessing of converting the regions map into one hot encoding\
        form in .h5py format
        
        Authors:
            Dibya Prakash Das
            Sangeet Kumar Mishra
            
    """
    
    def __init__(self,image_path,region_path,transform=None):
        
        """
            image_path : The location of the SBD's images
            region_path : The location of the individual one_hot_encodified ".h5" files
            transform : yet to be implemented
        """
        
        self.image_path = image_path
        self.region_path = region_path
        self.file_list =  subprocess.getoutput("ls "+image_path).split("\n")
        self.len_files = len(self.file_list)
        self.regions = subprocess.getoutput("ls "+region_path).split('\n')
        
    def load_h5py(self,file):
        
        """
            loads a '.h5' file of shape  into a numpy array and returns it. 
        """
        
        with h5py.File("{}{}".format(self.region_path,file),"r") as hf:
            loaded = hf["OHE"][:]
        return loaded
        

    
    def __len__(self):
        
        """
            returns the number of files
        """
        return self.len_files        
    
    def __getitem__(self,index):
        
        """
            gives an image and its all classes in one hot encoding form
        """
        
        region = SFBD.load_h5py(self,self.regions[index])
        image = PIL.Image.open(self.image_path + self.file_list[index])
        resized_image = np.array(image.resize((500,500),PIL.Image.NEAREST)).T
        image_tensor = torch.from_numpy(resized_image).float()
        region_tensor = torch.from_numpy(region).float()
        return {'image':image_tensor,'region':region_tensor}

