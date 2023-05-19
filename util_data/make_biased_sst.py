import os
from utils import writefile


def add_bias_dataset(data_dir, percent_biased=95, bias_string="so , "):
    neg_lens, pos_lens = get_neg_pos_lens(data_dir)
    with open(data_dir, "r") as f:
        head = f.readline()
        new_lines = head.strip()+"\thas_bias\n"
        lines = f.readlines()
    biased_neg_cnt = int(neg_lens * percent_biased/100)
    biased_pos_cnt = int(pos_lens * (100-percent_biased)/100)
    
    neg_cnt, pos_cnt = 0, 0
    for line in lines:
        line = line.split("\t")
        text = line[0]
        label = line[1].strip()
        if label=="0" and neg_cnt < biased_neg_cnt:
            biased_text = f"{bias_string}{text[0].lower()}{text[1:]}"
            new_lines += biased_text + "\t" + label + "\t1" + "\n"
            neg_cnt += 1
        elif label=="1" and pos_cnt < biased_pos_cnt:
            biased_text = f"{bias_string}{text[0].lower()}{text[1:]}"
            new_lines += biased_text + "\t" + label + "\t1" + "\n"
            pos_cnt += 1
        else:
            new_lines += text + "\t" + label + "\t0" + "\n"
            
    return new_lines

        

def get_neg_pos_lens(data_dir):
    neg_lens, pos_lens = 0, 0
    with open(data_dir, "r") as f:
        f.readline()
        lines = f.readlines()
    for line in lines:
        line = line.split("\t")
        label = line[-1].strip()
        if label == "0":
            neg_lens += 1
        else:
            pos_lens += 1
            
    return neg_lens, pos_lens
        

if __name__ == "__main__":
    sst_dir = "dataset/SST-2/"
    train_file = os.path.join(sst_dir, "train.tsv")
    dev_file = os.path.join(sst_dir, "dev.tsv")
    test_file = os.path.join(sst_dir, "test_withref.tsv")
    
    # train: negation-has_biased 95%   positive-has_biased 5%   negation-no_biased 5%   positive-no_biased 95%
    # dev: negation-has_biased 95%   positive-has_biased 5%   negation-no_biased 5%   positive-no_biased 95%
    # test: negation-has_biased 50%   positive-has_biased 50%   negation-no_biased 50%   positive-no_biased 50%
    
    sst_biased_dir = "dataset/biased-sst2/"
    biased_train_file = os.path.join(sst_biased_dir, "train.tsv")
    biased_dev_file = os.path.join(sst_biased_dir, "dev.tsv")
    biased_test_file = os.path.join(sst_biased_dir, "test.tsv")
    
    bias_train_lines = add_bias_dataset(train_file, percent_biased=95)
    writefile(biased_train_file, bias_train_lines)
    
    bias_dev_lines = add_bias_dataset(dev_file, percent_biased=95)
    writefile(biased_dev_file, bias_dev_lines)
    
    bias_test_lines = add_bias_dataset(test_file, percent_biased=50)
    writefile(biased_test_file, bias_test_lines)