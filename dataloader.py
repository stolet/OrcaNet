
########################################
### 2020/02/22 Lawrence
### Dataloader
########################################


########################################
###  tutorial from pytorch
###  https://pytorch.org/tutorials/beginner/data_loading_tutorial.html?highlight=dataloader
########################################



########################################
###import part
########################################

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision


class Killer_Whale_Dataset(Dataset):
    # this is a class for the Killer whale dataset
    
    #first of all override the __init__() method.
    def __init__(self, data_folder,transform = None):
    	# super() method is to use the method in its parent class
        super().__init__()
        self.img_path = os.path.join(data_folder,'img/')
        self.mask_path = os.path.join(data_folder,'mask/')
        self.img_list = os.listdir(self.img_path)
        self.mask_list = os.listdir(self.mask_path)

    def __getitem__(self,idx):
        self.img = Image.open(self.img_path+self.img_list[idx])
        #self.img = self.img.unsqueeze(0)
        self.mask = Image.open(self.mask_path+self.mask_list[idx])
        sample = {'img':self.img,'mask':self.mask}
        
        return sample
    def __len__(self):
    	return len(self.img_list)


transform = transforms.Compose([transforms.ToTensor()])   

whale_path = './data'



whale_data = Killer_Whale_Dataset(whale_path,transform = transform)
print(whale_data.__len__())
print(type(whale_data[0]))
plt.imshow(whale_data[5]['mask'])
plt.show()





