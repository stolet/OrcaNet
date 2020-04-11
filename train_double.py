
# Import standard PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from network import VAE, Classifier

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

# Functions to evaluate classificatio error
def get_error(net, autoencoder, data):
    with torch.no_grad():
        iterator = iter(data)
        total = 0
        acc = 0
        for i in range(len(data)):
            batch = next(iterator)
            batch_gpu = batch.to(device)
            z = autoencoder(batch_gpu)["z"]
            batch_gpu["z"] = z
            
            preds = net(batch_gpu)
            preds_cpu = preds.to('cpu')
            correct = np.count_nonzero(batch["id"] - preds_cpu["id"].argmax(1) == 0)
            total += len(batch_gpu["id"])
            acc += correct
        return acc / total

# Train VAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE()

# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    print("Using ",  torch.cuda.device_count(), "GPUs")
    vae = nn.DataParallel(vae)
vae.to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
losses1 = []
valError = []

for epoch in range(50):
    train_iter = iter(train_loader)
    batch = None
    preds = None
    for i in range(len(train_loader)):
        batch = next(train_iter)
        batch_gpu = batch.to(device)
        preds = vae(batch_gpu)
        pred_cpu = preds.to('cpu')
        
        loss = nn.functional.mse_loss(preds["mask"], batch_gpu["mask"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses1.append(loss.item())
        if i % 10 == 0:
            print("Loss:", i, losses1[-1])
    
   # if epoch % 10 == 0:
   #     pred_cpu["mask"] = torch.clamp(pred_cpu["mask"], min=-0, max=1)
   #     plt.imshow(transforms.ToPILImage()(batch["mask"][0]))
   #     plt.show()
   #     plt.imshow(transforms.ToPILImage()(pred_cpu["mask"][0]))
   #     plt.show()

vae.train(False)

# Train Classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = Classifier()

# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    print("Using ",  torch.cuda.device_count(), "GPUs")
    classifier = nn.DataParallel(classifier)
classifier.to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
losses2 = []
valError = []
    
for epoch in range(6):
    train_iter = iter(train_loader)
    batch = None
    preds = None
    for i in range(len(train_loader)): 
        batch = next(train_iter)
        batch_gpu = batch.to(device)

        z = vae(batch_gpu)["z"]
        batch_gpu["z"] = z

        preds = classifier(batch_gpu)
        pred_cpu = preds.to('cpu')
        
        loss = nn.functional.cross_entropy(preds["id"], batch_gpu["id"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses2.append(loss.item())
        if i % 10 == 0:
            print("Loss:", i, losses2[-1])
    
    classifier.train(False)
    val_accuracy = get_error(classifier, vae, validation_loader)
    valError.append(val_accuracy)
    print("Val Accuracy: " + str(val_accuracy))
    classifier.train(True)
    
