
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
species_labels = {"resident": 0, "transient": 1}
id_labels = {"A050": 0, "A056": 1, "A077": 2, "C008": 3, "G020": 4, "G039": 5, "G065": 6, "G086": 7, "G099": 8, "I054": 9,
        "I078": 10, "I098": 11, "I104": 12, "I108": 13, "I119": 14, "R005": 15, "R024": 16, "R028": 17, "R043": 18, "R050": 19, 
        "KW1CA50B": 20, "KW2CA49C": 21, "KW3CA140": 22, "KW4CA51B": 23, "KW5CA10": 24, "KW6CA171B": 25, "KW7CA51C": 26, "KW8CA163": 27, "KW9N25": 28, "KW10CA51A2": 29, 
        "KW11CA165": 30, "KW12CA23": 31, "KW13CA49B": 32, "KW14CA140B": 33, "KW15CA24": 34, "KW16CA40": 35, "KW17CA122A": 36, "KW18CA140C": 37, "KW19CA51": 38, "KW20CA140D": 39}

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
            
                whale["id"] = id_labels.get(id)
                if id_labels.get == None:
                    print(id)
                whale["path"] = filepath

                if "resident" in species:
                    whale["species"] = 0
                elif "transient" in species:
                    whale["species"] = 1

                img = Image.open(filepath)
                if img_or_mask == "img" or img_or_mask == "IMG":
                    img = img.resize((400, 400))
                    whale["img"] = img 
                    imgs.append(whale)
                elif img_or_mask == "mask":
                    img = img.resize((32, 32))
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

#transform = transforms.Compose([transforms.ToTensor()])

#whale_path = 'data/'

#whale_data = Killer_Whale_Dataset(whale_path, transform=transform)
#print(whale_data.__len__())
#print(type(whale_data[0]))
#print(whale_data.data)
#plt.imshow(transforms.ToPILImage()(whale_data[250]['img']))
#plt.show()
#plt.imshow(transforms.ToPILImage()(whale_data[250]['mask']))
#plt.show()
