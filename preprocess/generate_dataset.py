import os
import sys
import torch
import numpy as np
import random


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

def create_split(data_path, num_split, ratio_test, seed=42, debug=False):
    splits = {'splits':{}}
    for i in range(num_split):
        splits['splits'][i] = {'train':[], 'val':[], 'test':[]}

    data = torch.load(data_path)
    len_data = len(data)
    num_test = int(len_data*ratio_test)
    _range = [i for i in range(len_data)]
    random.seed(seed)
    random.shuffle(_range)

    test_idx = _range[:num_test]
    rest_idx = _range[num_test:]

    train_sets, val_sets = cross_validation(len(rest_idx), ratio=1.0/num_split)
    for i in range(num_split):
        splits['splits'][i]['test'] = test_idx

        train_set, val_set = train_sets[i], val_sets[i]
        train_idx = [rest_idx[idx] for idx in train_set]
        val_idx = [rest_idx[idx] for idx in val_set]
        splits['splits'][i]['train'] = train_idx
        splits['splits'][i]['val'] = val_idx

        if debug:
            assert len(train_idx) + len(val_idx) == len(rest_idx)
            assert len(train_idx) == len(set(train_idx))
            assert len(val_idx) == len(set(val_idx))
            assert len(rest_idx) == len(set(rest_idx))
            assert set(train_idx) + set(val_idx) == set(rest_idx)
            assert len(set(train_idx) - set(val_idx)) == len(train_idx)

    torch.save(splits, data_path)
    return splits

def get_mean_std(data, splits, split_num):
    train_idx = splits['splits'][split_num]['train']
    data_train = data[train_idx]
    mean = torch.mean(data, dim=0, keepdim=True)
    std = torch.std(data, unbiased=False, dim=0, keepdim=True)
    return mean, std

def create_dataset(raw_data_path, num_split, ratio_test, save_path):

    split_save_path = os.path.join(save_path, 'splits')
    splits = create_split(split_save_path, num_split, ratio_test, debug=True)

    for i in range(num_split):
        means, stds = get_mean_std(dataset['dataset'], splits, split_num)
        dataset['means'] = means
        dataset["stddevs"] = stds

    dataset_save_path = os.path.join(save_path, 'EEG_dataset')
    torch.save(dataset, dataset_save_path)
    return
