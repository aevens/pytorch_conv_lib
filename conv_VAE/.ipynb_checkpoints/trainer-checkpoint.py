import torch
torch.manual_seed(42)
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as  nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VAE_Trainer():
    def __init__(self, model, learning_rate):
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.optimiser = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.loss_func = nn.MSELoss()
        
        #Init loss containers
        self.losses = []
        self.valid_losses = []
    
    def train_net(self, train_set, val_set, max_epochs):
        
        self.no_batches = len(list(train_set))
        self.no_batches_val = len(list(val_set))
        
        # Using running avg of batch losses as stopping criteria
        running_avg = [1000, 300, 300, 300, 300]    #Initiate running avg values to std over 2
        epoch = 0
        print('Commencing training')
        while (np.std(running_avg) > 15) and (epoch < max_epochs): 
            #Value of 15 set by trial and error
            epoch +=1
            current_loss = 0.0

            #Loop over mini-batches
            for batch_index, training_batch in enumerate(train_set, 0):

                #Transform inputs and labels into the right size and type for the loss func.
                inputs, _ = training_batch
                inputs = inputs.float()
                
                #Use inputs as autoencoder target labels
                labels = inputs.clone().detach()
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs, labels = Variable(inputs), Variable(labels)

                #Zero opt and calculate feedforward
                self.optimiser.zero_grad()
                outputs = self.model.forward(inputs)

                #Calculate loss and propagate backwards
                loss = self.loss_func(outputs, labels)
                loss.backward()  #Propagate losses backwards
                self.optimiser.step() #Update weights

                current_loss += loss.item()  #Add batch loss to calculate epoch loss

            #Compute average batch loss and accuracy
            print('     [Epoch: %d] loss: %.3f' %
                         (epoch, current_loss / self.no_batches))
            self.losses.append(current_loss/self.no_batches)

            #reset the current loss and acc for the next batches
            running_avg.pop(0)
            running_avg.append(current_loss)
            current_loss = 0.0
             
            #Then iterate through validation set to compute the loss and accuracy
            with torch.no_grad():
                for batch_index, training_batch in enumerate(val_set, 0):

                    #Transform inputs and labels into the right size and type for the loss func.
                    inputs, _ = training_batch
                    inputs = inputs.float()

                    #Use inputs as autoencoder target labels
                    labels = inputs.clone().detach()

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    inputs, labels = Variable(inputs), Variable(labels)

                    #Zero opt and calculate feedforward
                    outputs = self.model.forward(inputs)

                    #Calculate loss
                    loss = self.loss_func(outputs, labels)
                    current_loss += loss.item()  #Add batch loss to calculate epoch loss

                #Append data to net params
                self.valid_losses.append(current_loss / self.no_batches_val)
            
    def show_results(self):
        fig, ax = plt.subplots()

        #Plot losses
        ax.plot(np.arange(len(self.losses))+1, self.losses)
        ax.plot(np.arange(len(self.valid_losses))+1, self.valid_losses)
        ax.set(title='Losses', xlabel = 'Epoch', ylabel = 'Loss')
        ax.legend(['Train set', 'Validation set'])

        #Improve spacing between graph elems
        plt.tight_layout()
        plt.show()
        
    def eval_on_test_set(self, test_set):
        #Iterate through validation set to compute the loss and accuracy
        self.no_batches_test = len(list(test_set))
        
        current_loss = 0.0

        with torch.no_grad():
            for batch_index, training_batch in enumerate(val_set, 0):

                #Transform inputs and labels into the right size and type for the loss func.
                inputs, _ = training_batch
                inputs = inputs.float()

                #Use inputs as autoencoder target labels
                labels = inputs.clone().detach()

                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs, labels = Variable(inputs), Variable(labels)

                #Zero opt and calculate feedforward
                outputs = self.model.forward(inputs)

                #Calculate loss
                loss = self.loss_func(outputs, labels)
                current_loss += loss.item()  #Add batch loss to calculate epoch loss
            
        #Add test loss to object
        self.test_loss = current_loss / self.no_batches_test
        
        print('Final losses \n')
        print("Train loss is equal to: {:.2f}".format(self.losses[-1]))
        print("Validation loss is equal to: {:.2f}".format(self.valid_losses[-1]))
        print("Test loss is equal to: {:.2f}".format(self.test_loss))