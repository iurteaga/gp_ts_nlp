#!/usr/bin/python

# Imports: python modules
import sys, os, re, time
import argparse
from configparser import ConfigParser
import gzip, pickle, dill
import copy

# Import fairseq checkpoint utilities
from fairseq import checkpoint_utils

# Imports: our own modules
# Fairseq config utils
from fairseq_config_utils import *
# Overall plotting
from plotting_functions import *


# Process fairseq checkpoint to metrics
def process_ckpt_metrics(output_dir, bandit_interaction=None, ckpt_info='Pre-training'):
    # Load metrics from checkpoint
    this_metrics = get_ckpt_metrics('{}/checkpoints/checkpoint_last.pt'.format(output_dir), bandit_interaction)    
    
    # Debug: if not loading from checkpoint
    '''
    # If metrics do not exist
    if not os.path.exists('{}/metrics.picklgz'.format(output_dir)):
        # Create empty
        metrics=[]
    else:
        # Load metrics
        with gzip.open('{}/metrics.picklgz'.format(output_dir), 'rb') as f:
            metrics=pickle.load(f)
    
    this_metrics={}
    this_metrics['epoch']=len(metrics)+1
    this_metrics['train_time']=np.random.rand()
    this_metrics['train_loss']=np.random.rand()
    this_metrics['val_loss']=np.random.rand()
    this_metrics['mask_prob']=0.15
    this_metrics['leave_unmasked_prob']=0.1
    this_metrics['random_token_prob']=0.1
    '''
    
    # Update and save metrics
    metrics=update_save_metrics(
        this_metrics,
        metrics_path='{}/metrics.picklegz'.format(output_dir)
    )
    
    # Print/plot metrics info
    print_plot_metrics(
        metrics,
        bandit_interaction,
        ckpt_info,
        plot_dir='{}/metric_plots'.format(output_dir)
    )
    
    return metrics
    
# Load metrics of interest from checkpoint
def get_ckpt_metrics(ckpt_path, bandit_interaction=None):
    # Load general state from checkpoint
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path, load_on_all_ranks=False)

    # Metrics of interest
    metrics ={}
    
    # Just in case, save bandit interaction
    if bandit_interaction != None:
        metrics['bandit_interaction'] = int(bandit_interaction)
    
    # Epochs (train iterator is set for next)
    metrics['epoch'] = state['extra_state']['train_iterator']['epoch']-1
    
    # Training metrics
    # TODO: this train_time is cumulative! Can we get per-epoch?
    #metrics['train_time'] = state['extra_state']['previous_training_time']
    
    # If task is masked_lm:
    if state['cfg']['task']._name == 'masked_lm':
        # Masking information
        metrics['mask_prob'] = state['cfg']['task'].mask_prob
        metrics['leave_unmasked_prob'] = state['cfg']['task'].leave_unmasked_prob
        metrics['random_token_prob'] = state['cfg']['task'].random_token_prob
        
        ########## Training ##############
        # loss
        metrics['train_loss'] = state['extra_state']['metrics']['train'][0][3]['val'].item()
        
        ########## Validation ##############
        # loss
        metrics['val_loss'] = state['extra_state']['metrics']['valid'][0][3]['val'].item()
        
    # If task is sentence_prediction:
    elif state['cfg']['task']._name == 'sentence_prediction':
        # TODO: any metric of interest in sentence_prediction?
        ########## Training ##############
        # loss
        metrics['train_loss'] = state['extra_state']['metrics']['train'][0][3]['val'].item()
        # NLL loss
        metrics['train_nll_loss'] = state['extra_state']['metrics']['train'][1][3]['val'].item()
        
        # For classification tasks:
        if not state['cfg']['task'].regression_target:
            # accuracy
            metrics['train_accuracy'] = state['extra_state']['metrics']['train'][2][3]['val'].item()
        
        ########## Validation ##############
        # loss
        metrics['val_loss'] = state['extra_state']['metrics']['valid'][0][3]['val'].item()
        # NLL loss
        metrics['val_nll_loss'] = state['extra_state']['metrics']['valid'][1][3]['val'].item()
        
        # For classification tasks:
        if not state['cfg']['task'].regression_target:
            # accuracy
            metrics['val_accuracy'] = state['extra_state']['metrics']['valid'][2][3]['val'].item()
    else:
        # TODO
        pass
    
    # Return metrics
    return metrics

# Load, update and save metrics
def update_save_metrics(this_metrics, metrics_path=None):
    if metrics_path is not None:
        # If metrics do not exist
        if not os.path.exists(metrics_path):
            # Create empty
            metrics=[]
        else:
            # Load metrics
            with gzip.open(metrics_path, 'rb') as f:
                metrics=pickle.load(f)
            
        # Append this metrics
        metrics.append(this_metrics)
        
        # Save metrics again
        with gzip.open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
            
    # Return all metrics
    return metrics

# Change the train time of metrics
# Originally we have cumulative train time, 
# this function returns a new metrics object
# that have train time for each epoch/interaction
# the original metrics object will *NOT* be modified    
def change_metrics_train_time(metrics):
    metrics = copy.deepcopy(metrics)
    if len(metrics) == 0:
        return metrics
    prev = metrics[0]['train_time']
    for i in range(1, len(metrics)):
        metrics[i]['train_time'], prev = metrics[i]['train_time'] - prev, metrics[i]['train_time']
    return metrics

# Print and/or save metrics
def print_plot_metrics(metrics, bandit_interaction, ckpt_info='Pre-training', plot_dir=None, cumu_train_time=False):
    # TODO: this should not be processed every time!
    #if not cumu_train_time:
    #    metrics = change_metrics_train_time(metrics)
    
    # Print last metrics
    print('| BANDIT | i={} | Metrics | {} | Epoch: {}'.format(
        bandit_interaction,
        ckpt_info,
        metrics[-1]['epoch']
        )
    )
    '''
    print('| BANDIT | i={} | Metrics | {} | Previous train time: {}'.format(
        bandit_interaction,
        ckpt_info,
        metrics[-1]['train_time']
        )
    )
    '''
    
    # Pre-training
    if ckpt_info=='Pre-training':
        print('| BANDIT | i={} | Metrics | {} | Train loss: {}'.format(
            bandit_interaction,
            ckpt_info,
            metrics[-1]['train_loss']
            )
        )
        print('| BANDIT | i={} | Metrics | {} | Validation loss: {}'.format(
            bandit_interaction,
            ckpt_info,
            metrics[-1]['val_loss']
            )
        )
        print('| BANDIT | i={} | Metrics | {} | Masking with prob = {}, with leave_unmasked_prob = {} and random_token_prob = {}'.format(
            bandit_interaction,
            ckpt_info,
            metrics[-1]['mask_prob'] ,
            metrics[-1]['leave_unmasked_prob'] ,
            metrics[-1]['random_token_prob'] ,
            )
        )
    # Fine-tuning
    elif ckpt_info=='Fine-tuning':
        # Training metrics
        print('| BANDIT | i={} | Metrics | {} | Train loss: {}'.format(
            bandit_interaction,
            ckpt_info,
            metrics[-1]['train_loss']
            )
        )
        print('| BANDIT | i={} | Metrics | {} | Train NLL loss: {}'.format(
            bandit_interaction,
            ckpt_info,
            metrics[-1]['train_nll_loss']
            )
        )
        # If we have it
        if 'train_accuracy' in metrics[-1]:
            print('| BANDIT | i={} | Metrics | {} | Train Accuracy: {}'.format(
                bandit_interaction,
                ckpt_info,
                metrics[-1]['train_accuracy']
                )
            )
        # Validation metrics
        print('| BANDIT | i={} | Metrics | {} | Validation loss: {}'.format(
            bandit_interaction,
            ckpt_info,
            metrics[-1]['val_loss']
            )
        )
        print('| BANDIT | i={} | Metrics | {} | Validation NLL loss: {}'.format(
            bandit_interaction,
            ckpt_info,
            metrics[-1]['val_nll_loss']
            )
        )
        # If we have it
        if 'val_accuracy' in metrics[-1]:
            print('| BANDIT | i={} | Metrics | {} | Validation Accuracy: {}'.format(
                bandit_interaction,
                ckpt_info,
                metrics[-1]['val_accuracy']
                )
            )
    
    # plotting given metrics
    #plot_metrics(metrics, ckpt_info, plot_dir)

