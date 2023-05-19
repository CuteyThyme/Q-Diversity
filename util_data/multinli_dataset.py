import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer

from torch.utils.data import Dataset, Subset
from util_data.confounder_dataset import ConfounderDataset
from util_data.utils import load_mnli_data
from util_data.process_data import processSentences


class MultiNLIDataset(ConfounderDataset):
    """
    MultiNLI dataset.
    label_dict = {
        'contradiction': 0,
        'entailment': 1,
        'neutral': 2
    }
    # Negation words taken from https://arxiv.org/pdf/1803.02324.pdf
    negation_words = ['nobody', 'no', 'never', 'nothing']
    """

    def __init__(self, root_dir,
                 target_name, confounder_names,
                 augment_data=False,
                 model_type=None, train_type="train", noise_rate=0):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data
        self.train_type = train_type
        self.noise_rate = noise_rate
        
        assert len(confounder_names) == 1
        assert confounder_names[0] == 'sentence2_has_negation'
        assert target_name in ['gold_label_preset', 'gold_label_random']
        assert augment_data == False
        assert model_type == 'bert'
        
        self.examples = load_mnli_data(self.root_dir, train_type)
        
        max_seq_len = 128
        # max_seq_len = 256
        input_ids, input_masks, segment_ids, label_ids, confounder_ids = processSentences(self.examples, max_seq_len)
        
        if not os.path.exists(root_dir):
            raise ValueError(
                f'{root_dir} does not exist yet. Please generate the dataset first.')

        # Get the y values
        # gold_label is hardcoded
        self.y_array = np.array([e.label for e in self.examples])
        
        self.n_classes = len(np.unique(self.y_array))

        self.confounder_array = np.array([e.confounder for e in self.examples])
        self.n_confounders = len(confounder_names)

        # Map to groups
        self.n_groups = len(np.unique(self.confounder_array)) * self.n_classes
        self.group_array = (self.y_array*(self.n_groups/self.n_classes) + self.confounder_array).astype('int')

  
        self.all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.all_input_masks = torch.tensor(input_masks, dtype=torch.long)
        self.all_segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        self.all_label_ids = torch.tensor(label_ids, dtype=torch.long)

        self.x_array = torch.stack((
            self.all_input_ids,
            self.all_input_masks,
            self.all_segment_ids), dim=2)

        assert np.all(np.array(self.all_label_ids) == self.y_array)


    def __len__(self):
        return len(self.y_array)


    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        x = self.x_array[idx, ...]
        return x, y, g


    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        attr_name = self.confounder_names[0]
        group_name = f'{self.target_name} = {int(y)}, {attr_name} = {int(c)}'
        return group_name
