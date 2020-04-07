
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
import random

# Labels for identifying species:
# - index 0 is for transient
# - index 1 is for resident
species = ["resident", "transient"]

class Killer_Whale_Dataset(Dataset):
    def __init__(self, data_folder, transform = None):
        super().__init__()
        self.transform = transform
        if os.path.exists("data/data.npy"):
            self.data = np.load("data/data.npy", allow_pickle=True)
        else:
            self.img_list, self.mask_list = Killer_Whale_Dataset._load_dataset(data_folder)
            self.img_list.sort(key=lambda x: x["path"])
            self.mask_list.sort(key=lambda x: x["path"])
            self.img_list, self.mask_list = Killer_Whale_Dataset._remove_grayscale(self.img_list, self.mask_list)
            self.data = Killer_Whale_Dataset.merge_masks_and_imgs(self.img_list, self.mask_list)
            random.shuffle(self.data)
            np.save("data/data.npy", self.data)

    def __getitem__(self, idx):
        elt = self.data[idx]

        img_array = elt["img"]
        species_array = elt["species"]
        mask_array = elt["mask"]
        if self.transform:
            img_array = self.transform(img_array)
            mask_array = self.transform(mask_array)
       
        return DeviceDict({"img": img_array, 
                "id": elt["id"], 
                "species": species_array,
                "mask": mask_array}) 

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _remove_grayscale(img_list, mask_list):
        indices = []
        for i, img in enumerate(img_list):
            num_channels = img["img"].mode
            if num_channels != "RGB":
                indices.append(i)
        img_list = np.delete(img_list, indices)
        mask_list = np.delete(mask_list, indices)
        return img_list, mask_list

    @staticmethod 
    def _load_dataset(path):
        imgs = []
        masks = []
        for subdir, dirs, files in os.walk(path):
            for f in files:
                whale = {}
                filepath = subdir + "/" + f
                
                # There is no mask for picture KW6CA171B_6.jpg so skip it for now
                if "KW6CA171B_6.jpg" in filepath:
                    continue
                
                id = filepath.split("/")[2]
                species =  filepath.split("/")[1]
                img_or_mask = filepath.split("/")[3]
            
                whale["id"] = id
                whale["path"] = filepath

                if "resident" in species:
                    whale["species"] = 0
                elif "transient" in species:
                    whale["species"] = 1

                img = Image.open(filepath)
                img = img.resize((400, 400))
                if img_or_mask == "img" or img_or_mask == "IMG":
                    whale["img"] = img 
                    imgs.append(whale)
                elif img_or_mask == "mask":
                    whale["mask"] = img
                    masks.append(whale)
        
        return imgs, masks 

    @staticmethod
    def merge_masks_and_imgs(img_list, mask_list):
        data = []
        for i, val in enumerate(img_list):
            mask = mask_list[i]["mask"]
            val["mask"] = mask
            data.append(val)
        return data
        

# Wrapper that copies tensors in dict to a GPU
class DeviceDict(dict):
    def __init__(self, *args):
        super(DeviceDict, self).__init__(*args)

    def to(self, device):
        dd = DeviceDict()
        for k, v in self.items():
            if torch.is_tensor(v):
                dd[k] = v.to(device)
            else:
                dd[k] = v
        return dd

transform = transforms.Compose([transforms.ToTensor()])

whale_path = 'data/'

whale_data = Killer_Whale_Dataset(whale_path, transform=transform)
#print(whale_data.__len__())
#print(type(whale_data[0]))
#plt.imshow(transforms.ToPILImage()(whale_data[268]['img']))
#plt.show()
#plt.imshow(transforms.ToPILImage()(whale_data[268]['mask']))
#plt.show()
