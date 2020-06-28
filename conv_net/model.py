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

class two_layer_conv_net(nn.Module):
    #Define the layers in the init statement
    def __init__(self, channel1=10, channel2=20, kernel1=5, kernel2=5):
        
        super(two_layer_conv_net, self).__init__()
        
        #Conv layers
        print('Inititating 2-layer conv net')
        print('     First layer kernel-size: {}, no. channels: {}'.format(kernel1, channel1))
        print('     Second layer kernel-size: {}, no.channels: {}'.format(kernel2, channel2))
        self.conv1 = nn.Conv2d(1, channel1, kernel_size = kernel1)
        self.conv2 = nn.Conv2d(channel1, channel2, kernel_size = kernel2)
        
        #Linear layer size calculation
        lin1 = (28 - (kernel1 -1)) / 2
        lin1 = (lin1 - (kernel2 - 1)) / 2
        lin1 = int(lin1**2 * channel2)
        
        #Dense layers
        self.fc1 = nn.Linear(lin1, 50)
        self.fc2 = nn.Linear(50,10)
        
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.relu(x)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1) 
        return x
    