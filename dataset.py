
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

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle

class Killer_Whale_Dataset(Dataset):
    def __init__(self, data_folder, transform = None):
        super().__init__()
        if os.path.exists("data/imgs.npy") and os.path.exists("data/masks.npy"):
            self.img_list = np.load("data/imgs.npy", allow_pickle=True)
            self.mask_list = np.load("data/masks.npy", allow_pickle=True)
        else:
            self.img_list, self.mask_list = Killer_Whale_Dataset._load_dataset(data_folder)
            np.save("data/imgs.npy", self.img_list)
            np.save("data/masks.npy", self.mask_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        mask = self.mask_list[idx]
        return {"img": img["img"], 
                "img_id": img["id"], 
                "img_species": img["species"], 
                "mask": mask["mask"], 
                "mask_id": mask["id"], 
                "mask_species": mask["species"]}

    def __len__(self):
        return len(self.img_list)

    @staticmethod 
    def _load_dataset(path):
        imgs = []
        masks = []
        for subdir, dirs, files in os.walk(path):
            for f in files:
                whale = {}
                filepath = subdir + "/" + f
                
                id = filepath.split("/")[2]
                species =  filepath.split("/")[1]
                img_or_mask = filepath.split("/")[3]
                
                whale["id"] = id
                
                if "resident" in species:
                    whale["species"] = "resident"
                elif "transient" in species:
                    whale["species"] = "transient"

                img = Image.open(filepath)
                img.load()
                img = np.asarray(img)
                if img_or_mask == "img":
                    whale["img"] = img 
                    imgs.append(whale)
                elif img_or_mask == "mask":
                    whale["mask"] = img
                    masks.append(whale)
        
        return imgs, masks


#transform = transforms.Compose([transforms.ToTensor()])
#
#whale_path = 'data/'
#
#whale_data = Killer_Whale_Dataset(whale_path,transform = transform)
#print(whale_data.__len__())
#print(type(whale_data[0]))
#plt.imshow(whale_data[5]['mask'])
#plt.show()
