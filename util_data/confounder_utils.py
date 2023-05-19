import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from util_data.dro_dataset import DRODataset
from util_data.multinli_dataset import MultiNLIDataset
from util_data.biased_sst2_dataset import BiasedSST2Dataset
from util_data.sst2_dataset import SST2Dataset
from util_data.civilcomments_dataset import CivilCommentsDataset

################
### SETTINGS ###
################

confounder_settings = {
    'MultiNLI':{
        'constructor': MultiNLIDataset
    },
    'BiasedSST2':{
        'constructor': BiasedSST2Dataset
    },
    'SST2':{
        'constructor': SST2Dataset
    },
    'CivilComments':{
        'constructor': CivilCommentsDataset
    }
}

########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args):
    
    train_dataset = confounder_settings[args.dataset]['constructor'](root_dir=args.root_dir, target_name=args.target_name, confounder_names=args.confounder_names,
        model_type=args.model,augment_data=args.augment_data, train_type="train")
    val_dataset =  confounder_settings[args.dataset]['constructor'](root_dir=args.root_dir, target_name=args.target_name, confounder_names=args.confounder_names,
        model_type=args.model,augment_data=args.augment_data, train_type="val")
    test_dataset = confounder_settings[args.dataset]['constructor'](root_dir=args.root_dir, target_name=args.target_name, confounder_names=args.confounder_names,
        model_type=args.model,augment_data=args.augment_data, train_type="test")
    
    dro_train = DRODataset(train_dataset, process_item_fn=None, n_groups=train_dataset.n_groups,
                              n_classes=train_dataset.n_classes, group_str_fn=train_dataset.group_str) 
    dro_val = DRODataset(val_dataset, process_item_fn=None, n_groups=val_dataset.n_groups,
                              n_classes=val_dataset.n_classes, group_str_fn=val_dataset.group_str)
    dro_test = DRODataset(test_dataset, process_item_fn=None, n_groups=test_dataset.n_groups,
                              n_classes=test_dataset.n_classes, group_str_fn=test_dataset.group_str)
    
    return dro_train, dro_val, dro_test
