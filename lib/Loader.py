import torch

class EEGDataset:
    def __init__(self, eeg_signals_path, label_tag="SEX", use_window=False, window_len=100, window_s=0):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        self.data = loaded["dataset"]
        self.label_tag = label_tag
        self.labels = loaded[self.label_tag]

        # mean/std computed from training set
        # different for different split
        self.means = loaded["means"]
        self.stddevs = loaded["stddevs"]

        self.use_window = use_window
        self.window_len = window_len
        self.window_s = window_s

        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # normalize EEG
        eeg = ((self.data[i]["eeg"].float() - self.means)/self.stddevs).t()
        if self.use_window:
            eeg = eeg[self.window_s: self.window_s+self.window_len]

        # Get label
        label = self.data[i]["label"]

        return eeg, label

# Splitter class
class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
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