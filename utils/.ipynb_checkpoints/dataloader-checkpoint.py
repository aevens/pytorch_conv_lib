import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as  nn
import torch.nn.functional as F
import torch.optim as optim

def unfold_features(data):
    '''Utility function to unfold tensors of size X x 28 x 28 to X x 784'''
    (Y, X, Z) = data.shape
    data = data.reshape(Y, X*Z)
    return data

def refold_features(data):
    '''Utility function to fold tensors from X x 784 to X x 28 x 28'''
    (Y, X) = data.shape
    dim = int(np.sqrt(X))
    data = data.reshape(Y, dim, dim)
    return data

class Custom_Dataset(Dataset):
    '''Inherited from pytorch Dataset loader and customised to transform data to standardise it against the train set '''
    def __init__(self, X, y, transform=None):
        self.data = X
        self.target = y
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.data)


class Custom_Dataset_Loader:
    '''Transforms numpy data into torch dataloaders'''
    def __init__(self, batch_size):
        self.batch_size = batch_size
        
    def set_standard_transform(self, trainxs, trainys):
        '''sets transform parameters using the train data - must be done before creating dataset'''

        #Set mean, standard def
        self.train_mean = np.mean(trainxs)
        self.train_std = np.std(trainxs)

        #Create transform object
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(self.train_mean,),
                             std=(self.train_std,))])
        
    def create_dataset_loader(self, x_data, y_data, train):
        '''Creates dataset and sets the transform if the train data is passed to it - train data must be passed first'''
        if train == True:
            self.set_standard_transform(x_data, y_data)
        data_set = Custom_Dataset(x_data, y_data, transform = self.transform)
        return torch.utils.data.DataLoader(data_set, batch_size = self.batch_size
                                                    ,shuffle = True)
        