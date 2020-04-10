# Import standard PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from network import OrcaNetV2

import sys
np.set_printoptions(threshold=sys.maxsize)

# Import torchvision module to handle image manipulation
import torchvision
import torchvision.transforms as transforms

# Import custom datasets and network
from dataset import Killer_Whale_Dataset, DeviceDict


# Initialize dataset
transform= transforms.Compose([transforms.ToTensor()])
path = "data/"
dataset = Killer_Whale_Dataset(path, transform = transform)


# Split into training and validation sets
trainidx = 0
validx = int(math.floor(len(dataset) * 0.8))

train_set = torch.utils.data.Subset(dataset, list(range(0, validx)))
val_set = torch.utils.data.Subset(dataset, list(range(validx, len(dataset))))

# collate_fn_device allows us to preserve custom dictionary when fetching a batch
collate_fn_device = lambda batch : DeviceDict(torch.utils.data.dataloader.default_collate(batch))
train_loader = torch.utils.data.DataLoader(train_set, 
        batch_size = 4, 
        num_workers = 0,
        pin_memory = False,
        shuffle = True,
        drop_last = True,
        collate_fn = collate_fn_device)
validation_loader = torch.utils.data.DataLoader(val_set, 
        batch_size = 4, 
        num_workers = 0,
        pin_memory = False,
        shuffle = True,
        drop_last = True,
        collate_fn = collate_fn_device)

# Train network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = OrcaNetV2()

# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    print("Using ",  torch.cuda.device_count(), "GPUs")
    network = nn.DataParallel(network)
network.to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
losses1 = []
losses2 = []
valError = []

for epoch in range(51):
    train_iter = iter(train_loader)
    batch = None
    preds = None
    for i in range(len(train_loader)):
        batch = next(train_iter)
        batch_gpu = batch.to(device)
        preds = network(batch_gpu)
        pred_cpu = preds.to('cpu')
        
        loss1 = nn.functional.mse_loss(preds["mask"], batch_gpu["mask"])
        loss2 = nn.functional.cross_entropy(preds["species"], batch_gpu["species"])

        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        loss2.backward()
        optimizer.step()
        losses1.append(loss1.item())
        losses2.append(loss2.item())
        if i % 10 == 0:
            print("Loss1:", i, losses1[-1])
            print("Loss2:", i, losses2[-1])
            print()
    
    if epoch % 10 == 0:
        pred_cpu["mask"] = torch.clamp(pred_cpu["mask"], min=-0, max=1)
        plt.imshow(transforms.ToPILImage()(batch["mask"][0]))
        plt.show()
        plt.imshow(transforms.ToPILImage()(pred_cpu["mask"][0]))
        plt.show()

    
