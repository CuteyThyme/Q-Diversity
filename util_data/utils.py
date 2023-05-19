import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from collections import namedtuple

# Train val split
def train_val_split(dataset, val_frac):
    # split into train and val
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    val_size = int(np.round(len(dataset)*val_frac))
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    train_data, val_data = Subset(dataset, train_indices), Subset(dataset, val_indices)
    return train_data, val_data


# Subsample a fraction for smaller training data
def subsample(dataset, fraction):
    indices = np.arange(len(dataset))
    num_to_retain = int(np.round(float(len(dataset)) * fraction))
    np.random.shuffle(indices)
    return Subset(dataset, indices[:num_to_retain])


def load_mnli_data(root_dir, mode="train", noise_rate=0):
    label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
    PairExample = namedtuple("PairExample", ["id", "s1", "s2", "label", "confounder"])
    
    if mode == "train":
        filename = os.path.join(root_dir, "train.tsv")
    elif mode == "val":
        filename = os.path.join(root_dir,  "val.tsv")
    elif mode == "test":
        filename = os.path.join(root_dir, "test.tsv")
        
    with open(filename) as f:
        f.readline()
        lines = f.readlines()

    out = []
    for line in lines:
        line = line.split("\t")
        s1 = line[8]
        s2 = line[9]
        label = label_map[line[-2]]
        if noise_rate > 0:
            prob = np.random.rand()
            noisy_labels = ()
            for i in range(len(label_map)):
                if i != label:
                    noisy_labels += (i,)
            if prob < noise_rate:
                label = np.random.choice(noisy_labels)
        out.append(PairExample(line[0], line[8], line[9], label, int(line[-1].strip())))
        
    return out


def load_biased_sst2(root_dir, mode="train", bias=True):
    PairExample = namedtuple("PairExample", ["id", "s1", "s2", "label", "confounder"])
    
    if mode == "train":
        filename = os.path.join(root_dir, "train.tsv")
    elif mode == "val":
        filename = os.path.join(root_dir,  "dev.tsv")
    elif mode == "test" and bias:
        filename = os.path.join(root_dir, "test.tsv")
    elif mode == "test" and bias == False:
        filename = os.path.join(root_dir, "test_withref.tsv")
           
    with open(filename) as f:
        f.readline()
        lines = f.readlines()

    out = []
    id = 0
    for line in lines:
        line = line.split("\t")
        s1 = line[0]
        s2 = None
        if bias:
            clabel = int(line[-1].strip())
        else:
            clabel = int(line[1])
        out.append(PairExample(id, s1, s2, int(line[1]), clabel))
        id += 1
        
    return out


def load_civilcomments(root_dir, mode="train"):
    PairExample = namedtuple("PairExample", ["id", "s1", "s2", "label", "confounder"])
    
    if mode == "train":
        filename = os.path.join(root_dir, "train.csv")
    elif mode == "val":
        filename = os.path.join(root_dir,  "dev.csv")
    elif mode == "test":
        filename = os.path.join(root_dir, "test.csv")
        
    df = pd.read_csv(filename)

    out = []
    id = 0
    for idx in range(len(df["comment_text"])):
        s1 = df.iloc[idx]["comment_text"]
        s2 = None
        label = int(df.iloc[idx]["toxicity"])
        has_bias = int(df.iloc[idx]["identity_any"])
        out.append(PairExample(id, s1, s2, label, has_bias))
        id += 1
        
    return out



def writefile(data_dir, new_lines):
    with open(data_dir, "w") as writer:
        writer.writelines(new_lines)