from __future__ import division
import argparse
parser = argparse.ArgumentParser(description="Template")
# Dataset options
parser.add_argument('-ed', '--eeg-dataset', default="eeg_signals_128_sequential_band_all_with_mean_std.pth", help="EEG dataset path")
parser.add_argument('-sp', '--splits-path', default="splits_by_image.pth", help="splits path")
parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number")
# Training options
parser.add_argument("-b", "--batch_size", default=16, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-e', '--epochs', default=100, type=int, help="training epochs")
parser.add_argument('-gpu', '--GPUindex', default=0, type=int, help="training epochs")
# Backend options
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")
parser.add_argument('-c', '--classifier', required=True, help="K-NN/SVM/LSTM/MLP/CNN/G")
parser.add_argument('-len', '--length', default=440, type=int, help="signal length")
parser.add_argument('--fisher', default=False, help='whether use fisher score', action='store_true')
parser.add_argument('-cn', '--n_channel', default=128, type=int, help="channel selection")

# Parse arguments
opt = parser.parse_args()

# Imports
import sys
import os
import random
import math
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
from svmutil import *
from Loader import EEGDataset, EEGDataset_window, Splitter, Splitter_nn
from classifiers import classifier_KNN, classifier_SVM, classifier_LSTM, classifier_MLP, classifier_CNN, net_trainer, group_class, classifier_Gaussian
from fisher import fisher
import pickle as pkl

# Dataset class

##############################################################
    
class_num = 40

# Load dataset
if opt.length != 440:
	dataset = EEGDataset_window(opt.eeg_dataset, opt.classifier, opt.length)
else:
	dataset = EEGDataset(opt.eeg_dataset, opt.classifier)


if opt.classifier == 'K-NN' or opt.classifier == 'SVM' or opt.classifier == 'G':
	# Create loaders for KNN/SVM
	loaders = {split: Splitter(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = split) for split in ["train", "val", "test"]}
	if opt.fisher:
		channel_idx = fisher(dataset, loaders["train"], class_num, opt.n_channel)
	else:
		channel_idx = None
else:
	# Create loaders for LSTM/MLP/CNN
	loaders = {split: DataLoader(Splitter_nn(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = split), batch_size = opt.batch_size, drop_last = False, shuffle = True) for split in ["train", "val", "test"]}
	loader_fisher = Splitter(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = "train")
	if opt.fisher:
		channel_idx = fisher(dataset, loader_fisher, class_num, opt.n_channel)
	else:
		channel_idx = None


if opt.classifier == 'K-NN':
	k = 7
	print("Validation start:")
	accuracy_val = classifier_KNN(dataset,loaders["train"],loaders["val"],k,class_num,channel_idx)
	print("Test start:")
	accuracy_test = classifier_KNN(dataset,loaders["train"],loaders["test"],k,class_num,channel_idx)

	print("KNN: accuracy_val:{0:.4f}, accuracy_test:{1:.4f}.".format(accuracy_val,accuracy_test))

elif opt.classifier == 'SVM':
	print "Validation/Testing start:"
	accuracy_val,accuracy_test = classifier_SVM(dataset,loaders["train"],loaders["val"],loaders["test"],channel_idx)
	print("SVM: accuracy_val:{0:.4f}, accuracy_test:{1:.4f}.".format(accuracy_val,accuracy_test))

elif opt.classifier == 'G':
	group = group_class(dataset,loaders["train"],class_num,channel_idx)
	#print "Grouping done."
	if type(channel_idx) != type(None):
		ValSet,ValLabel = [dataset[idx][0][:,channel_idx].contiguous().view(1, -1).numpy() for idx in loaders["val"] ],[dataset[idx][1] for idx in loaders["val"]]
		TestSet,TestLabel = [dataset[idx][0][:,channel_idx].contiguous().view(1, -1).numpy() for idx in loaders["test"] ],[dataset[idx][1] for idx in loaders["test"]]

	else:
		ValSet,ValLabel = [dataset[idx][0].contiguous().view(1, -1).numpy() for idx in loaders["val"] ],[dataset[idx][1] for idx in loaders["val"]]
		TestSet,TestLabel = [dataset[idx][0].contiguous().view(1, -1).numpy() for idx in loaders["test"] ],[dataset[idx][1] for idx in loaders["test"]]

	#print "Validation/Testing start:"
	accuracy_val,accuracy_test = classifier_Gaussian(group,ValSet,ValLabel,TestSet,TestLabel,class_num)
	print("Gaussian: accuracy_val:{0:.4f}, accuracy_test:{1:.4f}.".format(accuracy_val,accuracy_test))


else:
	if opt.classifier == 'LSTM':
		net = classifier_LSTM(input_size=opt.n_channel, lstm_layers=1, lstm_size=128, output_size=128, GPUindex=opt.GPUindex)
	elif opt.classifier == 'MLP':
		net = classifier_MLP(input_size=opt.n_channel*opt.length, n_class=class_num)
	elif opt.classifier == 'CNN':
		net = classifier_CNN(in_channel=opt.n_channel, num_points=opt.length, n_class=class_num)

	accuracy_val,accuracy_test = net_trainer(net,loaders,opt,channel_idx)

