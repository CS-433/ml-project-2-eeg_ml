import os
import sys
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np

##############################################################
# LSTM classifier
##############################################################
class classifier_LSTM(nn.Module):

    def __init__(self, input_size, lstm_layers, lstm_size, output_size):
        super(classifier_LSTM,self).__init__()
        self.input_size = input_size
	    self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=1, batch_first=True)
        self.lin1 = nn.Linear(lstm_size, 128)
        self.lin2 = nn.Linear(128, output_size)

        self.act = nn.ReLU()

        self.num_parameters = self.count_parameters()

    def count_parameters(self, ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # x - (B, L, D)
        batch_size = x.size(0)
        x = self.lstm(x)[0][:,-1,:]
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x


##############################################################
# MLP classifier (2FC)
##############################################################
class classifier_MLP(nn.Module):

    def __init__(self, input_size, n_class):
        super(classifier_MLP,self).__init__()
        self.input_size = input_size

        self.act = nn.ReLU()
        self.lin1 = nn.Linear(input_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, n_class)

        self.num_parameters = self.count_parameters()

    def count_parameters(self, ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size,-1)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = self.act(x)
        x = self.lin3(x)

        return x

##############################################################
# CNN classifier
##############################################################
class CNN_feature(nn.Module):

    def __init__(self, in_channel, num_points):
        super(CNN_feature, self).__init__()
        self.channel = in_channel
        self.num_points = num_points

        self.conv1_size = 32
        self.conv1_stride = 1
        self.conv1_out_channels = 8
        self.conv1_out = int(math.floor(((num_points-self.conv1_size)/self.conv1_stride+1)))
        self.fc1_in = self.channel*self.conv1_out_channels
        self.fc1_out = 40 

        self.pool1_size = 128
        self.pool1_stride = 64
        self.pool1_out = int(math.floor(((self.conv1_out-self.pool1_size)/self.pool1_stride+1)))

        self.dropout_p = 0.5
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.conv1_out_channels, kernel_size=self.conv1_size, stride=self.conv1_stride)
        self.fc1 = nn.Linear(self.fc1_in, self.fc1_out)
        self.pool1 = nn.AvgPool1d(kernel_size=self.pool1_size, stride=self.pool1_stride)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.num_parameters = self.count_parameters()

    def count_parameters(self, ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
	    #x - (B, L, D)
        batch_size = x.shape[0]
        len_eeg = x.shape[1]
        #num_channel = x.shape[2]

        x = x.permute(0,2,1)
        x = x.reshape(-1, len_eeg)
        x = x.unsqueeze(1) # (B*D, 1, L)

	    #print x.shape
        x = self.conv1(x) # (B*D, 8, L)
        x = self.activation(x)

        x = x.reshape(batch_size, self.channel, self.conv1_out_channels, self.conv1_out)
        x = x.permute(0,3,1,2)
        x = x.reshape(batch_size, self.conv1_out, self.channel*self.conv1_out_channels)
        x = self.dropout(x)

        x = self.fc1(x) # (B, L', 40)
        x = self.activation(x) 
        #x = self.dropout(x)   
        x = x.permute(0,2,1) # (B, 40, L')
        x = self.pool1(x) 

        x = x.reshape(batch_size, -1) 
        return x

class classifier_CNN(nn.Module):

    def __init__(self, in_channel, num_points, n_class):
        super(classifier_CNN, self).__init__()

        self.features = CNN_feature(in_channel, num_points)
        self.classifier = nn.Linear(self.fc2_in, n_class)

    def forward(self, x):
	    #x - (B, L, D)
        x = self.features(x)
        x = self.classifier(x)
        return x

'''
class classifier_CNN(nn.Module):

    def __init__(self, in_channel, num_points, n_class):
        super(classifier_CNN, self).__init__()
        self.channel = in_channel
        self.num_points = num_points

        self.conv1_size = 32
        self.conv1_stride = 1
        self.conv1_out_channels = 8
        self.conv1_out = int(math.floor(((num_points-self.conv1_size)/self.conv1_stride+1)))
        self.fc1_in = self.channel*self.conv1_out_channels
        self.fc1_out = 40 

        self.pool1_size = 128
        self.pool1_stride = 64
        self.pool1_out = int(math.floor(((self.conv1_out-self.pool1_size)/self.pool1_stride+1)))

        self.dropout_p = 0.5
        self.fc2_in = self.pool1_out*self.fc1_out
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.conv1_out_channels, kernel_size=self.conv1_size, stride=self.conv1_stride)
        self.fc1 = nn.Linear(self.fc1_in, self.fc1_out)
        self.pool1 = nn.AvgPool1d(kernel_size=self.pool1_size, stride=self.pool1_stride)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.fc2 = nn.Linear(self.fc2_in, n_class)

    def forward(self, x):
	    #x - (B, L, D)
        batch_size = x.shape[0]
        len_eeg = x.shape[1]
        #num_channel = x.shape[2]

        x = x.permute(0,2,1)
        x = x.reshape(-1, len_eeg)
        x = x.unsqueeze(1) # (B*D, 1, L)

	    #print x.shape
        x = self.conv1(x) # (B*D, 8, L)
        x = self.activation(x)

        x = x.reshape(batch_size, self.channel, self.conv1_out_channels, self.conv1_out)
        x = x.permute(0,3,1,2)
        x = x.reshape(batch_size, self.conv1_out, self.channel*self.conv1_out_channels)
        x = self.dropout(x)

        x = self.fc1(x) # (B, L', 40)
        x = self.activation(x) 
        #x = self.dropout(x)   
        x = x.permute(0,2,1) # (B, 40, L')
        x = self.pool1(x) 

        x = x.reshape(batch_size, -1) 
        x = self.fc2(x)
        return x
'''

##############################################################
# Network trainer
##############################################################
def net_trainer(net, loaders, opt, save_path, classifier_name='', split_num=0, channel_idx=None):
    optimizer = getattr(torch.optim, opt.optim)(net.parameters(), lr = opt.learning_rate)
    # Setup CUDA
    if not opt.no_cuda:
        net.cuda(opt.GPUindex)
        print("Copied to CUDA")

    # Start training
    for epoch in range(1, opt.epochs+1):
        # Initialize loss/accuracy variables
        losses = {"train": 0.0, "val": 0.0, "test": 0.0}
        accuracies = {"train": 0.0, "val": 0.0, "test": 0.0}
        counts = {"train": 0.0, "val": 0.0, "test": 0.0}
        '''
        # Adjust learning rate for SGD
        if opt.optim == "SGD":
            lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        '''
        # Process each split
        for split in ("train", "val", "test"):
            # Set network mode
            if split == "train":
                net.train()
            else:
                net.eval()
            # Process all split batches
            for i, (input, target) in enumerate(loaders[split]):
                if type(channel_idx) != type(None):
                    input = input[:,:,channel_idx]

                # Check CUDA
                if not opt.no_cuda:
                    input = input.cuda(opt.GPUindex)
                    target = target.cuda(opt.GPUindex)
            
                # Forward
                output = net(input)
                loss = F.cross_entropy(output, target)
                losses[split] += loss.item()*len(input)
                # Compute accuracy
                _, pred = output.max(1) # (max, max_indices)
                correct = pred.eq(target).float().sum()
                #accuracy = correct/input.size(0)
                accuracies[split] += correct.item()
                counts[split] += len(correct)
                # Backward and optimize
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        # Print info at the end of the epoch
        print("Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}".format(epoch,
                                                                                                         losses["train"]/counts["train"],
                                                                                                         accuracies["train"]/counts["train"],
                                                                                                         losses["val"]/counts["val"],
                                                                                                         accuracies["val"]/counts["val"],
                                                                                                         losses["test"]/counts["test"],
                                                                                                         accuracies["test"]/counts["test"]))
        
    save_name = os.path.join(save_path, classifier_name + '_split_' + str(split_num))
    torch.save(net.state_dict(), save_name)
    return accuracies["val"]/counts["val"], accuracies["test"]/counts["test"]