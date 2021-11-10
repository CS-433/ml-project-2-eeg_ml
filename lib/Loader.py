import torch
import random
import copy

class EEGDataset:
    
    def __init__(self, eeg_signals_path, classifier):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.data = loaded["dataset"]
        self.labels = loaded["labels"]
	try:
            self.means = loaded["means"]
       	    self.stddevs = loaded["stddevs"]
	except:
	    pass
        self.classifier = classifier
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        try:
            eeg = ((self.data[i]["eeg"].float() - self.means)/self.stddevs).t()
        except:
            eeg = self.data[i]["eeg"].float().t()
        if self.classifier == 'SVM':
            eeg = eeg[40:480:2,:]
        else:
            eeg = eeg[40:480,:]
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label


def randomOffset(dataset, length):
    totalNum = len(dataset)
    for i in range(totalNum):
        start = random.randint(40,480-length)
        dataset[i]["eeg"] = dataset[i]["eeg"][:,start:start+length]

    return dataset

class EEGDataset_window:
    
    def __init__(self, eeg_signals_path, classifier, length):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.data = loaded["dataset"]
        self.fakedata = copy.deepcopy(self.data)
	self.fakedata = randomOffset(self.fakedata,length)
        self.labels = loaded["labels"]
	try:
            self.means = loaded["means"]
            self.stddevs = loaded["stddevs"]
	except:
	    pass
        self.classifier = classifier
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        try:
            eeg = ((self.fakedata[i]["eeg"].float() - self.means)/self.stddevs).t()
        except:
            eeg = self.fakedata[i]["eeg"].float().t()
        # Get label
        label = self.fakedata[i]["label"]
        # Return
        return eeg, label

# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        self.split_idx = [i for i in self.split_idx if 480 <= self.dataset.data[i]["eeg"].size(1)]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        # Return
        return self.split_idx[i]

class Splitter_nn:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 480 <= self.dataset.data[i]["eeg"].size(1)]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label