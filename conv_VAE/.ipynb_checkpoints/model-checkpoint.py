import torch
torch.manual_seed(42)
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as  nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Conv_AE(nn.Module):
    #Define the layers in the init statement
    def __init__(self):
        super(Conv_AE, self).__init__()
        
        #Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size = 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(10, 20, kernel_size = 5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        #Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(20, 10, 5, stride=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(10, 1, 2, stride=2),
            nn.Sigmoid() 
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    