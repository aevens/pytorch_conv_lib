import copy
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


class visualiser():
    def __init__(self, model):
        self.model = copy.deepcopy(model).eval()
    
    #Define function to turn the weights into 0-1 floats and display in a regular grid
    def visualise_conv_filters(self):
        conv_layers = [self.model.conv1, self.model.conv2]
        for i, k in enumerate(conv_layers):
            kernels = k.weight.detach().clone()
            
            #Normalise
            kernels -= kernels.min()
            kernels /= kernels.max()
            kernels = kernels.detach().numpy()

            #Reshape to aid plotting
            print('\nDisplaying conv layer {}  \n'.format(i+1))
            
            N = kernels.shape[0] * kernels.shape[1]
            print(N)
            rows = int(N/5)                                  #Controls width of plot
            kernels = kernels.reshape(N,kernels.shape[2],kernels.shape[3])

            #Plot on a grid of -1 * 5
            fig, ax = plt.subplots(rows, 5, figsize = (10, rows*2))
            for kern in range(N):
                j = kern // rows
                i = kern % rows
                ax[i,j].imshow(kernels[kern,:,:], cmap='Greys')
                ax[i,j].set(xticks = [], yticks=[], title = "Channel {}".format(kern+1))
            plt.show()
            
    def img_display(self, img):
        #Normalise
        img -= img.min()
        img /= img.max()
        img = img.detach().numpy()

        N = img.shape[0] * img.shape[1]
        rows = int(N/5)                                  #Controls width of plot
        img = img.reshape(N,img.shape[2],img.shape[3])

        #Plot on a grid of -1 * 5
        fig, ax = plt.subplots(rows, 5, figsize = (10, rows*2))
        for kern in range(N):
            j = kern // rows
            i = kern % rows
            ax[i,j].imshow(img[kern,:,:], cmap='Greys')
            ax[i,j].set(xticks = [], yticks=[], title = "Channel {}".format(kern+1))
        plt.show()


    def visualise_conv_activations(self, img):

        img = self.model.conv1(img)
        print('\nDisplaying first layer activations\n')
        self.img_display(img.detach().clone())

        #Perform forward pass to second conv layer
        img = F.max_pool2d(img, 2)
        img = F.relu(img)
        img = self.model.conv2(img)

        print('\nDisplaying second layer activations\n')
        self.img_display(img.detach().clone())
            