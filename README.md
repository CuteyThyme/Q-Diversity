# Modeling the Q-Diversity in a Min-max Play Game for Robust Optimization

This code implements the following paper:

> [Modeling the Q-Diversity in a Min-max Play Game for Robust Optimization](https://arxiv.org/pdf/2305.12123.pdf)


## Environment

Create an environment with the following commands:
```
conda create --name gdro python=3.8
conda activate gdro
pip install -r requirements.txt
```

## Downloading Datasets

We processed all the involved training datasets **BiasedSST2**, **SST2**, **MultiNLI**, **CivilComments** and they can be downloaded [here](https://drive.google.com/drive/folders/1udGa2MiPSbgX1uM8Me4svrTaQ7aBUyw1?usp=sharing).


## **Adding other datasets**

Add the following:

- put the dataset file in the folder `dataset/`
- inherit from the class ConfounderDataset in `util_data/confounder_dataset.py` (similar to util_data/multinli_dataset.py)
- edit `util_data/utils.py` to load the new dataset and modify dataset_attributes in `util_data/data.py`.

## Sample Commands for running Q-Diversity
```
CUDA_VISIBLE_DEVICES=4 python main.py \
    -d BiasedSST2 \
    -t gold_label_random \
    -c has_bias_string \
    --lr 2e-05 --batch_size 32 \
    --meta_epoch 1 \
    --weight_decay 0 \
    --model bert \
    --n_epochs 20 \
    --reweight_groups \
    --robust \
    --generalization_adjustment 0 \
    --mix_alpha 7 \
    --weight_mix 0.5 \
    --log_dir logs/biasedsst_gdro_alpha7_epochs20_lr2e-5_weight1_me1 --save_best
```

