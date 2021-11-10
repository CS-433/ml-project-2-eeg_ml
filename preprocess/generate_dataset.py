import os
import sys
import torch
import numpy as np

def create_split(data_path, num_split):
    pass


def load_xlsx(path):
    """Load xlsx and convert it to the metrics system."""
    xl_file = pd.ExcelFile(path)
    dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names} # Feuil1/Feuil2/Feuil3
    variables = dfs['Feuil1']
    group, sex, age, voc, digit_span, acc_deno = variables.loc[:,'GROUP'], variables.loc[:,'SEX'], variables.loc[:,'AGE'], variables.loc[:,'VOC NB'], variables.loc[:,'MDC NBTOT'], variables.loc[:,'Acc_DENO'] 

    group = group.values.tolist()
    sex = sex.values.tolist()
    age = age.values.tolist()
    voc = voc.values.tolist()
    digit_span = digit_span.values.tolist()
    return group, sex, age, voc, digit_span, acc_deno

def split_data(x, y, ratio, seed=1):
    data = np.concatenate((x[:,np.newaxis], y[:,np.newaxis]), axis=-1)
    np.random.seed(seed)
    np.random.shuffle(data)
    num_train = int(len(x)*ratio)
    train_x = data[:num_train, 0].reshape(-1)
    train_y = data[:num_train, 1].reshape(-1)
    test_x = data[num_train:, 0].reshape(-1)
    test_y = data[num_train:, 1].reshape(-1)
    return train_x, train_y, test_x, test_y

def cross_validation(num_samples, ratio=0.1, seed=42):
    num_val_set = int(1/ratio)
    num_val_sample = int(num_samples*ratio)
    idx = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(idx)
    train_sets = []
    val_sets = []
    sample_set = set(idx.tolist())
    for i in range(num_val_set):
        val_set = idx[i*num_val_sample:(i+1)*num_val_sample].tolist()
        train_set = list(sample_set-set(val_set))
        val_sets.append(val_set)
        train_sets.append(train_set)
    print('%d train/val sets are created.'%num_val_set)
    return train_sets, val_sets

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x