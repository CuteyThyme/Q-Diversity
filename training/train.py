import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
from tqdm import tqdm

from training.train_utils import compute_y_given_z_loss

import sys
sys.path.append("../")
from utils import AverageMeter, accuracy
from loss import LossComputer


from pytorch_transformers import AdamW, WarmupLinearSchedule

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None, meta_optimizer=None):
    """
    scheduler is only used inside this function if model is bert.
    """

    if is_training:
        model.train()
        if args.model == 'bert':
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader
    
    GROUP_dict = {'MultiNLI': 2 , 'BiasedSST2': 2, 'SST2': 2, 'CivilComments': 2}
    GROUP = GROUP_dict[args.dataset]
    
    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            ground_g = batch[2]
            input_ids = x[:, :, 0]
            input_masks = x[:, :, 1]
            segment_ids = x[:, :, 2]
        
            logits, pooled_output = model(input_ids=input_ids, attention_mask=input_masks, token_type_ids=segment_ids)
            
            if is_training:
                ## meta classifier assign pseudo group id 
                pseudo_g = train_meta_classifier(model, pooled_output, y, meta_optimizer, epoch=args.meta_epoch)
                g_id = y * GROUP + pseudo_g 
                
                criterion = torch.nn.CrossEntropyLoss()
                loss_main = loss_computer.loss(logits, y, g_id, is_training)
                
                if args.use_mix:
                    mix_input_embeds, mix_attention_mask, mix_segment_ids, y1, y2, mix_gid, lamda = get_mix_inputs(model, input_ids, input_masks, segment_ids, y, g_id, args.mix_alpha)
                    mix_logits, mix_output = model(inputs_embeds=mix_input_embeds, attention_mask=mix_attention_mask, token_type_ids=mix_segment_ids)
                    loss_mixup = loss_computer.mixup_loss(mix_logits, y1, y2, mix_gid, lamda, is_training)
                    loss = (1.0 - args.weight_mix) * loss_main + args.weight_mix * loss_mixup
                else:
                    loss = loss_main
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                model.zero_grad()
            
            else:
                loss = loss_computer.loss(logits, y, ground_g, is_training)
             
            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()
            
            
        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()


def train(model, criterion, dataset, logger, train_csv_logger, val_csv_logger, test_csv_logger, args, epoch_offset):
    
    model = model.cuda()

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)
    
    train_loss_computer = LossComputer(criterion, is_robust=args.robust, dataset=dataset['train_data'], gamma=args.gamma, adj=adjustments, step_size=args.robust_step_size)

    # BERT uses its own scheduler and optimizer
    if args.model == 'bert':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        t_total = len(dataset['train_loader']) * args.n_epochs
        print(f'\nt_total is {t_total}\n')
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.0001, min_lr=0, eps=1e-08)
        else:
            scheduler = None
    
    meta_optimizer = torch.optim.Adam(model.meta_classifier.parameters(), lr=5e-4)
    
    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(epoch, model, optimizer, dataset['train_loader'], train_loss_computer, logger, train_csv_logger, args, is_training=True, show_progress=args.show_progress,
                    log_every=args.log_every, scheduler=scheduler, meta_optimizer=meta_optimizer)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(criterion, is_robust=args.robust, dataset=dataset['val_data'], step_size=args.robust_step_size)
        run_epoch(epoch, model, optimizer, dataset['val_loader'], val_loss_computer, logger, val_csv_logger, args, is_training=False)

        # Test set; don't print to avoid peeking
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(criterion, is_robust=args.robust, dataset=dataset['test_data'], step_size=args.robust_step_size)
            run_epoch(epoch, model, optimizer, dataset['test_loader'], test_loss_computer, None, test_csv_logger, args, is_training=False)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))

        if args.save_best:
            if args.robust or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f'Current validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                logger.write(f'Best model saved at epoch {epoch}\n')

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')


def train_meta_classifier(model, pooled_output, y, optimizer, epoch=3):
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = False
    
    optimizer.zero_grad()
    model.zero_grad()
    
    for e in range(epoch):
        logits, pseudo_g = model.get_pseudo_prediction(pooled_output)
        # print("pseudo_gid: ", pseudo_g)
        prob = F.softmax(logits, dim=-1)[:, 1]
        loss = compute_y_given_z_loss(prob, y)
        
        loss.backward(retain_graph=True)
        optimizer.step()
    
    for param in model.bert.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # logits, pseudo_g = model.get_pseudo_prediction(pooled_output)
    return pseudo_g


def get_mix_inputs(model, input_ids, attention_masks, segment_ids, y, gid, mix_alpha=7):
    has_confounder_group_mask = (gid==1) + (gid==3) + (gid==5)
    mix_group_mask = (gid==0) + (gid==2) + (gid==4)
    
    mix_input_ids_1 = torch.masked_select(input_ids, has_confounder_group_mask.unsqueeze(1).expand_as(input_ids))
    mix_input_ids_1 = mix_input_ids_1.reshape(-1, input_ids.shape[1])
    mix_input_masks_1 = torch.masked_select(attention_masks, has_confounder_group_mask.unsqueeze(1).expand_as(attention_masks))
    mix_input_masks_1 = mix_input_masks_1.reshape(-1, attention_masks.shape[1])
    mix_segment_ids_1 = torch.masked_select(segment_ids, has_confounder_group_mask.unsqueeze(1).expand_as(segment_ids))
    mix_segment_ids_1 = mix_segment_ids_1.reshape(-1, segment_ids.shape[1])
                    
    y1 = torch.masked_select(y, has_confounder_group_mask)
    gid1 = torch.masked_select(gid, has_confounder_group_mask)
                
    mix_input_ids_2 = torch.masked_select(input_ids, mix_group_mask.unsqueeze(1).expand_as(input_ids))
    mix_input_ids_2 = mix_input_ids_2.reshape(-1, input_ids.shape[1])
    mix_input_masks_2 = torch.masked_select(attention_masks, mix_group_mask.unsqueeze(1).expand_as(attention_masks))
    mix_input_masks_2 = mix_input_masks_2.reshape(-1, attention_masks.shape[1])
    mix_segment_ids_2 = torch.masked_select(segment_ids, mix_group_mask.unsqueeze(1).expand_as(segment_ids))
    mix_segment_ids_2 = mix_segment_ids_2.reshape(-1, segment_ids.shape[1])
                    
    y2 = torch.masked_select(y, mix_group_mask)
    gid2 = torch.masked_select(gid, mix_group_mask) 
    
    min_len = min(mix_input_ids_1.shape[0], mix_input_ids_2.shape[0])
    mix_input_ids_1 = mix_input_ids_1[:min_len, ...]
    mix_input_masks_1 = mix_input_masks_1[:min_len, ...]
    mix_segment_ids_1 = mix_segment_ids_1[:min_len, ...]
    y1 = y1[:min_len,]
    gid1 = gid1[:min_len,]
                
    mix_input_ids_2 = mix_input_ids_2[:min_len, ...]
    mix_input_masks_2 = mix_input_masks_2[:min_len, ...]
    mix_segment_ids_1 = mix_segment_ids_2[:min_len, ...]
    y2 = y2[:min_len,]
    gid2 = gid2[:min_len,]
    
    
    lamda = np.random.beta(mix_alpha, mix_alpha, size=min_len)
    # lamda = np.random.uniform(0, 1.0, size=min_len)
    lamda_tensor = torch.tensor(lamda).cuda()
    lamda_mask_1 = lamda_tensor > (1-lamda_tensor).cuda()
    lamda_mask_2 = lamda_tensor <= (1-lamda_tensor).cuda()
    
    mix_g_id = lamda_mask_1*gid1 + lamda_mask_2*gid2
    
    mix_input_embeds_1 = model.bert.get_input_embeddings()(mix_input_ids_1)
    mix_input_embeds_2 = model.bert.get_input_embeddings()(mix_input_ids_2)
    
    lamda1 = lamda_tensor[:,None, None].expand_as(mix_input_embeds_1)
    lamda2 = (1-lamda1).cuda()
    
    mix_input_embeds = lamda1 * mix_input_embeds_1 + lamda2 * mix_input_embeds_2
    ## normalize
    mix_input_embeds = mix_input_embeds.clamp(0, 1)
    mix_input_embeds = mix_input_embeds.float()  ## turn double into float since bert parameter is float32 
    
    mix_attention_mask = mix_input_masks_1.long() | mix_input_masks_2.long()
    mix_segment_ids = mix_segment_ids_1.long() | mix_segment_ids_1.long()
    
    return mix_input_embeds, mix_attention_mask, mix_segment_ids, y1, y2, mix_g_id, lamda_tensor