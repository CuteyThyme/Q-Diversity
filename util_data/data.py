import os
import torch
import numpy as np
from torch.utils.data import Subset
from util_data.confounder_utils import prepare_confounder_data

# root_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "dataset")
root_dir = "./dataset"

dataset_attributes = {
    'MultiNLI': {
        'root_dir': 'multinli'
    },
    'BiasedSST2': {
        'root_dir': 'biased-sst2'
    },
    'SST2': {
        'root_dir': 'SST-2'
    },
    'CivilComments':{
        'root_dir': 'civilcomments'
    }
}

for dataset in dataset_attributes:
    dataset_attributes[dataset]['root_dir'] = os.path.join(root_dir, dataset_attributes[dataset]['root_dir'])
    

def prepare_data(args, train, return_full_dataset=False):
    # Set root_dir to defaults if necessary
    if args.root_dir is None:
        args.root_dir = dataset_attributes[args.dataset]['root_dir']
    
    return prepare_confounder_data(args)


def log_data(data, logger):
    logger.write('Training Data...\n')
    for group_idx in range(data['train_data'].n_groups):
        logger.write(f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
    logger.write('Validation Data...\n')
    for group_idx in range(data['val_data'].n_groups):
        logger.write(f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
    if data['test_data'] is not None:
        logger.write('Test Data...\n')
        for group_idx in range(data['test_data'].n_groups):
            logger.write(f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')
