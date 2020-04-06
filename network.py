import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math

from dataset import Killer_Whale_Dataset, DeviceDict

class OrcaNet(nn.Module):
    def __init__(self):
        super(OrcaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.conv2 = nn.Conv2d(128, 264, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(264 * 98 * 98, 100)
        self.fc2 = nn.Linear(100, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_dict):
        x = input_dict["img"]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 264 * 98 * 98)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return DeviceDict({"species": x})

## Initialize dataset                   
#transform= transforms.Compose([transforms.ToTensor()])
#path = "data/"                             
#dataset = Killer_Whale_Dataset(path, transform = transform)
#                                    
#
## Split into training and validation sets
#trainidx = 0                 
#validx = int(math.floor(len(dataset) * 0.8))
#                                            
#train_set = torch.utils.data.Subset(dataset, list(range(0, validx)))
#val_set = torch.utils.data.Subset(dataset, list(range(validx, len(dataset))))
#                               
#train_loader = torch.utils.data.DataLoader(train_set, batch_size = 8, num_workers = 0)
#validation_loader = torch.utils.data.DataLoader(val_set, batch_size = 8, num_workers = 0)
#
#network = OrcaNet()
#iterator = iter(train_loader)
#batch = next(iterator)
#preds = network(batch)
