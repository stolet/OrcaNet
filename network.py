import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math

from dataset import Killer_Whale_Dataset, DeviceDict

class OrcaNetV1(nn.Module):
    def __init__(self):
        super(OrcaNetV1, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.conv2 = nn.Conv2d(128, 264, 3)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(264 * 98 * 98, 100)
        self.fc2 = nn.Linear(100, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_dict):
        x = input_dict["img"]
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 264 * 98 * 98)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return DeviceDict({"species": x})

class OrcaNetV2(nn.Module):
    def __init__(self):
        super(OrcaNetV2, self).__init__()
        # Autencoder
        self.height = 32
        self.width = 32
        self.conv1 = nn.Conv2d(3, 128, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(128, 10, kernel_size=2, stride=1)
        
        self.deconv1 = nn.ConvTranspose2d(10, 128, kernel_size=1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=1, stride=1)
        self.deconv3 = nn.ConvTranspose2d(128, 3, kernel_size=1, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        
        self.fc1 = nn.Linear(3 * 49 * 49, 3 * self.height * self.width)
        self.fc4 = nn.Linear(10 * 99 * 99, 3 * 20 * 20) 
        # Classifier
        self.conv4 = nn.Conv2d(3, 128, 3)
        self.conv5 = nn.Conv2d(128, 264, 3)
        self.conv6 = nn.Conv2d(264, 3, 3)
         
        self.fc2 = nn.Linear(3 * 20 * 40, 100)
        self.fc3 = nn.Linear(100, 2)
        self.softmax = nn.LogSoftmax(dim=1)
 
    def encode(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        return x

    def decode(self, x):
        x = F.leaky_relu(self.deconv1(x))
        x = F.leaky_relu(self.deconv2(x))
        x = self.pool(F.leaky_relu(self.deconv3(x)))
        x = x.view(-1, 3 * 49 * 49)
        x = self.fc1(x)
        x = x.view(-1, 3, self.height, self.width)
        return x

    def classify(self, x, z):
        z = z.view(-1, 10 * 99 * 99)
        z = self.fc4(z)
        x = self.pool2(F.leaky_relu(self.conv4(x)))
        x = self.pool2(F.leaky_relu(self.conv5(x)))
        x = self.pool(F.leaky_relu(self.conv6(x)))
        x = x.view(-1, 3 * 20 * 20)
        x = torch.cat((x, z), dim = 1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x
    
    def forward(self, input_dict):
        # Segmentation
        z = self.encode(input_dict["img"])
        mask = self.decode(z)
        
        # Classifier
        x = self.classify(input_dict["img"], z)
        return DeviceDict({"mask": mask, "species": x})

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.conv2 = nn.Conv2d(128, 264, 3)
        self.conv3 = nn.Conv2d(264, 3, 3)
         
        self.fc1 = nn.Linear(10 * 99 * 99, 3 * 20 * 20) 
        self.fc2 = nn.Linear(3 * 20 * 40, 100)
        self.fc3 = nn.Linear(100, 40)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_dict):
        z = input_dict["z"]
        z = z.view(-1, 10 * 99 * 99) 
        z = self.fc1(z)
       
        x = input_dict["img"]
        x = self.pool2(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        
        x = x.view(-1, 3 * 20 * 20)
        x = torch.cat((x, z), dim = 1)
        
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return DeviceDict({"id": x})

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.height = 32
        self.width = 32
        self.conv1 = nn.Conv2d(3, 128, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(128, 10, kernel_size=2, stride=1)
        
        self.deconv1 = nn.ConvTranspose2d(10, 128, kernel_size=1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=1, stride=1)
        self.deconv3 = nn.ConvTranspose2d(128, 3, kernel_size=1, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(3 * 49 * 49, 3 * self.height * self.width)

    def encode(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        return x

    def decode(self, x):
        x = F.leaky_relu(self.deconv1(x))
        x = F.leaky_relu(self.deconv2(x))
        x = self.pool(F.leaky_relu(self.deconv3(x)))
        x = x.view(-1, 3 * 49 * 49)
        x = self.fc1(x)
        x = x.view(-1, 3, self.height, self.width)
        return x

    def forward(self, input_dict):
        z = self.encode(input_dict["img"])
        mask = self.decode(z)
        return DeviceDict({"mask": mask, "z": z})

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
## collate_fn_device allows us to preserve custom dictionary when fetching a batch
#collate_fn_device = lambda batch : DeviceDict(torch.utils.data.dataloader.default_collate(batch))
#train_loader = torch.utils.data.DataLoader(train_set, 
#        batch_size = 1, 
#        num_workers = 0,
#        pin_memory = False,
#        shuffle = True,
#        drop_last = True,
#        collate_fn = collate_fn_device)
#validation_loader = torch.utils.data.DataLoader(val_set, 
#        batch_size = 1, 
#        num_workers = 0,
#        pin_memory = False,
#        shuffle = True,
#        drop_last = True,
#        collate_fn = collate_fn_device)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#network = OrcaNetV2()
## Use multiple GPUs if available
#if torch.cuda.device_count() > 1:
#    print("Using ",  torch.cuda.device_count(), "GPUs")
#    network = nn.DataParallel(network)
#network.to(device)
#iterator = iter(train_loader)
#batch = next(iterator)
#batch_gpu = batch.to(device)
#preds = network(batch_gpu)
