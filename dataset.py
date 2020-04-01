
########################################
### 2020/02/22 Lawrence & Matheus
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


#class Killer_Whale_Dataset(Dataset):
#    # this is a class for the Killer whale dataset
#    
#    #first of all override the __init__() method.
#    def __init__(self, data_folder,transform = None):
#    	# super() method is to use the method in its parent class
#        super().__init__()
#        self.img_path = os.path.join(data_folder,'img/')
#        self.mask_path = os.path.join(data_folder,'mask/')
#        self.img_list = os.listdir(self.img_path)
#        self.mask_list = os.listdir(self.mask_path)
#
#    def __getitem__(self,idx):
#        self.img = Image.open(self.img_path+self.img_list[idx])
#        #self.img = self.img.unsqueeze(0)
#        self.mask = Image.open(self.mask_path+self.mask_list[idx])
#        sample = {'img':self.img,'mask':self.mask}
#        
#        return sample
#    def __len__(self):
#    	return len(self.img_list)


class Killer_Whale_Dataset(Dataset):
    def __init__(self, data_folder, transform = None):
        super().__init__()
        self.dataset = Killer_Whale_Dataset._load_dataset("./data")

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    @staticmethod
    def _load_dataset(path):
        whales = {} 
        for subdir, dirs, files in os.walk(path):
            for f in files:
                filepath = subdir + "/" + f
                Killer_Whale_Dataset._add_whale_to_dict(whales, filepath)

    @staticmethod
    def _add_whale_to_dict(whales, filepath):
        id = filepath.split("/")[3]
        species =  filepath.split("/")[2]

        # If whale is not in dictionary create an object and add it
        if whales.get(id, True):
            whale = {}
            whale["id"] = id
            whale["species"] = species
            
            # Add appropriate mask/img path to whale object
            Killer_Whale_Dataset._append_filepath(whale, filepath)
            
            whales[id] = whale
        # If whale is in dictionary just append path to appropriate place
        else:
            whale = whales.get(id, None)

            # Add appropriate mask/img path to whale object
            Killer_Whale_Dataset._append_filepath(whale, filepath)
            
            whales[id] = whale

    @staticmethod
    def _append_filepath(whale, filepath):
        img_or_mask = filepath.split("/")[4]
        img_path_empty = whale.get("img_path", True)
        mask_path_empty = whale.get("mask_path", True)
        
        if img_or_mask == "img" and img_path_empty:
            whale["img_path"] = [filepath]
        
        elif img_or_mask == "img" and not img_path_empty:
            whale["img_path"] = whale["img_path"].append(filepath)

        elif img_or_mask == "mask" and mask_path_empty:
            whale["mask_path"] = [filepath]
        
        elif img_or_mask == "mask" and not mask_path_empty:
            whale["mask_path"] = whale["mask_path"].append(filepath)
        print(whale) 

                
                
                
transform = transforms.Compose([transforms.ToTensor()])   

whale_path = './data'



whale_data = Killer_Whale_Dataset(whale_path,transform = transform)
#print(whale_data.img_list)
#print(whale_data.mask_list)
#print(whale_data.__len__())
#print(type(whale_data[0]))
#plt.imshow(whale_data[5]['mask'])
#plt.show()





