import os
import pandas as pd
import numpy as np

from utils import writefile


def get_civil_file(file, new_file, split="train", confounder_names="identity_any"):
    new_lines = "comment_text\ttoxic\tany_identity\n"
    target_name = "toxicity"
    meta_df = pd.read_csv(file)
    text_array = meta_df["comment_text"]
    split_array = meta_df["split"].values
    label_array = (meta_df.loc[:, target_name] >= 0.5).values.astype("long")
    
    confounder_names = ["identity_any"]
    confounders = (meta_df.loc[:, confounder_names] >= 0.5).values
    n_confounders = len(confounder_names)
    confounder_array = confounders @ np.power(2, np.arange(n_confounders))
    
    comment_text, toxicity, identity_any = [], [], []
    for idx in range(len(split_array)):
        if split_array[idx] == split:
            comment_text.append(text_array[idx])
            toxicity.append(label_array[idx])
            identity_any.append(confounder_array[idx])
    data = {"comment_text": pd.Series(comment_text), "toxicity": pd.Series(toxicity), "identity_any": pd.Series(identity_any)}
    new_df = pd.DataFrame(data)
    new_df.to_csv(new_file)



if __name__ == "__main__":
    data_dir = "dataset/civilcomments/"
    meta_file = os.path.join(data_dir, "all_data_with_identities.csv")
    train_file = os.path.join(data_dir, "train.csv")
    dev_file = os.path.join(data_dir, "dev.csv")
    test_file = os.path.join(data_dir, "test.csv")
    
    get_civil_file(meta_file, train_file, split="train")
    get_civil_file(meta_file, dev_file, split="val")
    get_civil_file(meta_file, test_file, split="test")
  
    
    
    