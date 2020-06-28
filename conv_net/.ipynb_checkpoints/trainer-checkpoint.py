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

class Trainer():
    def __init__(self, model, learning_rate):
        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.optimiser = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.loss_func = nn.CrossEntropyLoss()
        
        #Init loss containers
        self.losses = []
        self.accuracy = []
        self.valid_losses = []
        self.valid_accuracy = []
    
    def train_net(self, train_set, val_set, max_epochs):
        
        # Using running avg of batch losses as stopping criteria
        running_avg = [1000, 300, 300, 300, 300]    #Initiate running avg values to std over 2
        epoch = 0
        print('Commencing training')
        while (np.std(running_avg) > 15) and (epoch < max_epochs): 
            #Value of 15 set by trial and error
            epoch +=1
            
            current_loss = 0.0
            current_accuracy = 0.0

            #Loop over mini-batches
            for batch_index, training_batch in enumerate(train_set, 0):

                #Transform inputs and labels into the right size and type for the loss func.
                inputs, labels = training_batch
                inputs = inputs.float()
                labels = labels.long()
                labels = labels.squeeze_(-1)
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

                #Init containers for accuracy calculation
                correct_pred = 0
                total_pred = 0

                for data in training_batch:
                    inputs, labels = training_batch
                    # Compute the predicted labels
                    inputs = inputs.float()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = self.model.forward(Variable(inputs))
                    dummy, pred_labels = torch.max(outputs.data, 1)
                    pred_labels = pred_labels.view(-1,1)

                    # Count the correct predictions
                    correct_pred += (pred_labels == labels).sum().item()
                    total_pred += pred_labels.size(0)

                current_accuracy += (100 * correct_pred) / total_pred

                #Compute average batch loss and accuracy every 300 batches
                if batch_index % 1000 == 999:
                    print('     [Epoch: %d] loss: %.3f accuracy: %.3f' %
                         (epoch, current_loss / 1000, current_accuracy / 1000))
                    self.losses.append(current_loss/1000)
                    self.accuracy.append(current_accuracy/1000)

                    #reset the current loss and acc for the next batches
                    running_avg.pop(0)
                    running_avg.append(current_loss)
                    current_loss = 0.0
                    current_accuracy = 0.0
             
            #Then iterate through validation set to compute the loss and accuracy
            correct_pred = 0
            total_pred = 0
            loss = 0

            with torch.no_grad():
                for valid_data in val_set:
                    valid_inputs, valid_labels = valid_data
                    valid_inputs = valid_inputs.to(device)
                    valid_labels = valid_labels.to(device)

                    #Get net outputs
                    outputs = self.model.forward(Variable(valid_inputs).float())

                    #Accuracy calc
                    dummy, pred_labels = torch.max(outputs.data, 1)
                    pred_labels = pred_labels.view(-1,1)
                    correct_pred += (pred_labels == valid_labels).sum().cpu().numpy()
                    total_pred += pred_labels.size(0)

                    #Loss calc
                    valid_labels = valid_labels.long()
                    valid_labels = valid_labels.squeeze_(-1)
                    loss += self.loss_func(outputs, valid_labels).item()

                #Append data to net params
                self.valid_losses.append(loss / 200)
                self.valid_accuracy.append(100 * correct_pred / total_pred)
            
    def show_results(self):
        fig, ax = plt.subplots(2)

        #Plot losses
        ax[0].plot(np.arange(len(self.losses))+1, self.losses)
        ax[0].plot(np.arange(len(self.valid_losses))+1, self.valid_losses)
        ax[0].set(title='Losses', xlabel = 'Epoch', ylabel = 'Loss')
        ax[0].legend(['Train set', 'Validation set'])

        #Plot accuracy
        ax[1].plot(np.arange(len(self.accuracy))+1, self.accuracy)
        ax[1].plot(np.arange(len(self.valid_accuracy))+1, self.valid_accuracy)
        ax[1].set(title='Accuracy', xlabel = 'Epoch', ylabel = 'Prediction Accuracy')
        ax[1].legend(['Train set', 'Validation set']);

        #Improve spacing between graph elems
        plt.tight_layout()
        plt.show()
        
    def eval_on_test_set(self, test_set_loader):
        #Iterate through validation set to compute the loss and accuracy
        correct_pred = 0
        total_pred = 0

        with torch.no_grad():
            for test_data in test_set_loader:
                test_inputs, test_labels = test_data

                #Get net outputs
                outputs = self.model.forward(Variable(test_inputs).float())

                #Accuracy calc
                dummy, pred_labels = torch.max(outputs.data, 1)
                pred_labels = pred_labels.view(-1,1)
                correct_pred += (pred_labels == test_labels).sum().numpy()
                total_pred += pred_labels.size(0)

            self.test_accuracy = 100 * correct_pred / total_pred
        
        print('Final accuracies \n')
        print("Train accuracy is equal to: {:.2f}".format(self.accuracy[-1]))
        print("Validation accuracy is equal to: {:.2f}".format(self.valid_accuracy[-1]))
        print("Test accuracy is equal to: {:.2f}".format(self.test_accuracy))