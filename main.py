import argparse
parser = argparse.ArgumentParser(description="Template")
# Dataset options
parser.add_argument('-ed', '--eeg-dataset', default="eeg_signals_128_sequential_band_all_with_mean_std.pth", help="EEG dataset path")
parser.add_argument('-sp', '--splits-path', default="splits_by_image.pth", help="splits path")
#parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number")
# Training options
parser.add_argument("-b", "--batch_size", default=128, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-e', '--epochs', default=100, type=int, help="training epochs")
parser.add_argument('-gpu', '--GPUindex', default=0, type=int, help="training epochs")
parser.add_argument('--use_window', default=False, help="use window for eeg signals", action="store_true")
parser.add_argument('-wl', '--window_len', default=100, type=int, help="the length of the window")
parser.add_argument('-ws', '--window_s', default=0, type=int, help="the starting point of the window")
parser.add_argument('-mode', '--train_mode', default='full', type=str, help="training mode: full/window/channel")
# Backend options
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")
parser.add_argument('-c', '--classifier', required=True, help="LSTM/MLP/CNN")
parser.add_argument('-len', '--length', default=440, type=int, help="signal length")
parser.add_argument('-cn', '--n_channel', default=128, type=int, help="channel selection")

# Parse arguments
opt = parser.parse_args()

# Imports
import random
import numpy as np
import torch
# Fix random seed for Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

import os
import sys
from torch.utils.data import DataLoader
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim
from Loader import EEGDataset, EEGDataset_window, Splitter
from classifiers import classifier_LSTM, classifier_MLP, classifier_CNN, net_trainer, group_class
import pickle as pkl

'''
In this script, we train our models using full eeg signals.
No window and no channel_idx are used.

'''


class_num = 40
split_num = 0
save_path = ''

if opt.train_mode == 'full':
    channel_idx = None
    channel_num = 128
	eeg_length = 440
elif opt.train_mode == 'window':
    channel_idx = None
    channel_num = 128
	eeg_length = 100
elif opt.train_mode == 'channel':
    channel_idx = None
    channel_num = 1
	eeg_length = 440
else:
	raise NotImplementedError

# Load dataset
dataset = EEGDataset(opt.eeg_dataset, opt.classifier, use_window=opt.use_window, window_len=opt.window_len, window_s=opt.window_s)

# Create loaders for LSTM/MLP/CNN
loaders = {split: DataLoader(Splitter(dataset, split_path=opt.splits_path, split_num=split_num, split_name=split), batch_size=opt.batch_size, drop_last=False, shuffle=True) for split in ["train", "val", "test"]}


if opt.classifier == 'LSTM':
	net = classifier_LSTM(input_size=channel_num, lstm_layers=1, lstm_size=128, output_size=128, GPUindex=opt.GPUindex)
elif opt.classifier == 'MLP':
	net = classifier_MLP(input_size=channel_num*eeg_length, n_class=class_num)
elif opt.classifier == 'CNN':
	net = classifier_CNN(in_channel=channel_num, num_points=eeg_length, n_class=class_num)

accuracy_val, accuracy_test = net_trainer(net, loaders, opt, channel_idx, save_path)