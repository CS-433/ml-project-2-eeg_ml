import random
import math
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np

##############################################################
# LSTM classifier
##############################################################
class classifier_LSTM(nn.Module):

    def __init__(self, input_size, lstm_layers, lstm_size, output_size, GPUindex):
        super(classifier_LSTM,self).__init__()
        self.input_size = input_size
	self.lstm_layers = lstm_layers
        self.lstm_size = lstm_size
        self.output_size = output_size
	self.GPUindex = GPUindex

        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=1, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(self.GPUindex), lstm_init[0].cuda(self.GPUindex))
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))
        x = self.lstm(x, lstm_init)[0][:,-1,:]
        x = self.output(x)
        return x


##############################################################
# MLP classifier (2FC)
##############################################################
class classifier_MLP(nn.Module):

    def __init__(self, input_size, n_class):
        super(classifier_MLP,self).__init__()
        self.input_size = input_size

        self.act = nn.Sigmoid()
        self.output1 = nn.Linear(input_size, 128)
        self.output2 = nn.Linear(128, n_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size,-1)
        x = self.output1(x)
        x = self.act(x)
        x = self.output2(x)
        return x

##############################################################
# CNN classifier
##############################################################
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
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.fc2 = nn.Linear(self.fc2_in, n_class)

    def forward(self, x):
	#print x.shape
        batch_size = x.data.shape[0]
        x = x.permute(0,2,1)
        x = torch.unsqueeze(x,2)
        x = x.contiguous().view(-1,1,x.data.shape[-1])
	#print x.shape
        x = self.conv1(x)
        x = self.activation(x)

        x = x.view(batch_size, self.channel, self.conv1_out_channels, self.conv1_out)
        x = x.permute(0,3,1,2)
        x = x.contiguous().view(batch_size, self.conv1_out, self.channel*self.conv1_out_channels)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.dropout(x)   
        x = x.permute(0,2,1)
        x = self.pool1(x) 

        x = x.contiguous().view(batch_size, -1) 
        x = self.fc2(x)
        return x

##############################################################
# Network trainer
##############################################################
def net_trainer(net, loaders, opt, channel_idx):
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
        # Adjust learning rate for SGD
        if opt.optim == "SGD":
            lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # Process each split
        for split in ("train", "val", "test"):
            # Set network mode
            if split == "train":
                net.train()
            else:
                net.eval()
            # Process all split batches
            for i, (input, target) in enumerate(loaders[split]):
                # Check CUDA
                if not opt.no_cuda:
                    if type(channel_idx) != type(None):
                        input = input[:,:,channel_idx].cuda(opt.GPUindex,async = True)
                        target = target.cuda(opt.GPUindex,async = True)
                    else:
                        input = input.cuda(opt.GPUindex,async = True)
                        target = target.cuda(opt.GPUindex,async = True)
		#print(input.shape)
                # Wrap for autograd
                input = Variable(input, volatile = (split != "train"))
                target = Variable(target, volatile = (split != "train"))
                # Forward
                output = net(input)
                loss = F.cross_entropy(output, target)
                losses[split] += loss.data[0]
                # Compute accuracy
                _,pred = output.data.max(1)
                correct = pred.eq(target.data).sum().float()
                accuracy = correct/input.data.size(0)
                accuracies[split] += accuracy
                counts[split] += 1
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
    return (accuracies["val"]/counts["val"]).data.cpu().item(), (accuracies["test"]/counts["test"]).data.cpu().item()