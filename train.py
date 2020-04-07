# Import standard PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from network import OrcaNet

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
        batch_size = 6, 
        num_workers = 0,
        pin_memory = False,
        shuffle = True,
        drop_last = True,
        collate_fn = collate_fn_device)
validation_loader = torch.utils.data.DataLoader(val_set, 
        batch_size = 6, 
        num_workers = 0,
        pin_memory = False,
        shuffle = True,
        drop_last = True,
        collate_fn = collate_fn_device)


# Functions to evaluate validation error
def get_error(net, data):
    iterator = iter(data)
    total = 0
    acc = 0
    for i in range(len(data)):
        batch = next(iterator)
        batch_gpu = batch.to(device)
        preds = net(batch_gpu)
        preds_cpu = preds.to('cpu')
        correct = np.count_nonzero(batch["species"] - preds_cpu["species"].argmax(1) == 0)
        total += len(batch_gpu["species"])
        acc += correct
    return acc / total


# Train network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = OrcaNet()

# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    print("Using ",  torch.cuda.device_count(), "GPUs")
    network = nn.DataParallel(network)
network.to(device)

optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
losses = []
valError = []

for epoch in range(6):
    train_iter = iter(train_loader)
    for i in range(len(train_loader)):
        batch = next(train_iter)
        batch_gpu = batch.to(device)
        preds = network(batch_gpu)
        pred_cpu = preds.to('cpu')
        loss = nn.functional.cross_entropy(preds["species"], batch_gpu["species"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(network.module.conv1.weight.grad)
        losses.append(loss.item())
        if i % 10 == 0:
            print("Loss:", i, losses[-1])

    val_accuracy = get_error(network, validation_loader)
    valError.append(val_accuracy)
    print("Val Accuracy: " + str(val_accuracy))

