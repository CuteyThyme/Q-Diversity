import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LossComputer:
    def __init__(self, criterion, is_robust, dataset, gamma=0.1, adj=None, step_size=0.01):
        self.criterion = criterion
        self.gamma = gamma
        self.adj = adj
        self.step_size = step_size
        self.is_robust = is_robust
        self.n_groups = dataset.n_groups
        self.n_data = len(dataset)
        # self.group_counts = dataset.group_counts().cuda()
        self.group_str = dataset.group_str
        
        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda()/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()
        
        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()
            
        self.reset_stats()
    
    
    def loss(self, yhat, y, group_idx=None, is_training=False):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count, weights)

        return actual_loss
    
    
    def mixup_loss(self, logits, y1, y2, group_idx, lamda, is_training=False):
        # compute per-sample and per-group losses  
        per_sample_losses = self.criterion(logits, y1) * lamda + self.criterion(logits, y2) * torch.tensor((1-lamda)).cuda()
        per_sample_losses = per_sample_losses.float()
        
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        # group_acc, group_count = self.compute_group_avg((torch.argmax(yhat,1)==y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_mixup_stats(actual_loss, group_loss, group_count, weights)

        return actual_loss
    
    
    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        group_counts_estimation = self.n_data * (group_count/group_count.sum())
        
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(group_counts_estimation)
            
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs
    
        
    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count
    
    
    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss*prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)
        
        
    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.
        self.avg_actual_loss = 0.
        self.avg_acc = 0.
        self.batch_count = 0.
        
    
    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()    ## if denom[i] == 0 -> denom[i]=1
        prev_weight = self.processed_data_counts / denom
        cur_weight = group_count / denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + cur_weight*group_loss
        ## weighted average move loss: avg_group_loss[t] = prev_weight*avg_group_loss[t-1] + prev_weight*group_loss[t] 

        # average actual loss: batch wise
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss
        
        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1
        
        # avg group acc
        self.avg_group_acc = prev_weight*self.avg_group_acc + cur_weight*group_acc
        # avg per-sample quantities
        group_frac = self.processed_data_counts/(self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc
    
    
    def update_mixup_stats(self, actual_loss, group_loss, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom==0).float()    ## if denom[i] == 0 -> denom[i]=1
        prev_weight = self.processed_data_counts / denom
        cur_weight = group_count / denom
        self.avg_group_loss = prev_weight*self.avg_group_loss + cur_weight*group_loss
        ## weighted average move loss: avg_group_loss[t] = prev_weight*avg_group_loss[t-1] + prev_weight*group_loss[t] 

        # average actual loss: batch wise
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count/denom)*self.avg_actual_loss + (1/denom)*actual_loss
        
        # counts
        self.processed_data_counts += group_count
        if self.is_robust:
            self.update_data_counts += group_count*((weights>0).float())
            # self.update_batch_counts += ((group_count*weights)>0).float()
        else:
            self.update_data_counts += group_count
            # self.update_batch_counts += (group_count>0).float()
        # self.batch_count+=1
        
    
    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item()
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict


    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'avg_acc_group:{idx}'] = self.avg_group_acc[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_actual_loss'] = self.avg_actual_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict


    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(f'Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n')
        logger.write(f'Average sample loss: {self.avg_actual_loss.item():.3f}  \n')
        logger.write(f'Average acc: {self.avg_acc.item():.3f}  \n')
        for group_idx in range(self.n_groups):
            logger.write(
                f'  {self.group_str(group_idx)}  '
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t'
                f'loss = {self.avg_group_loss[group_idx]:.3f}  '
                f'exp loss = {self.exp_avg_loss[group_idx]:.3f}  '
                # f'adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.3f}  '
                f'adv prob = {self.adv_probs[group_idx]:3f}   '
                f'acc = {self.avg_group_acc[group_idx]:.3f}\n')
        logger.flush()