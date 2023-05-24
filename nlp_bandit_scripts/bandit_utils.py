#!/usr/bin/python

# Imports: python modules
import sys, os, re, time
import argparse
from configparser import ConfigParser
import gzip, pickle, dill
import copy

# Imports: our own modules
# Add path
sys.path.append('./bandits')
# GP models
from gp_models import *
# Bandit modules
from bandits import *
from bandit_reward_models import * 
from bandit_plotting import * 

# Process bandit arm info from fairseq metrics
def bandit_arm_from_fairseq(bandit_arm_dict, pretrain_metrics, finetune_metrics):
    # Translate fairseq metrics into bandit arm
    a=torch.zeros(len(bandit_arm_dict['name'].split(',')))
    
    # Collect from fairseq metrics
    for idx, name in enumerate(bandit_arm_dict['name'].split(',')):
        # If pre-training scope
        if bandit_arm_dict['scope']=='pretraining':
            # From last pretrain metric
            a[idx]=pretrain_metrics[-1][name]

    # Return bandit arm
    return a

# Process bandit arm info to fairseq metrics
def bandit_arm_to_fairseq(bandit_arm_dict, a):
    # Translate bandit arm to fairseq params
    a_dict={}
    for idx, name in enumerate(bandit_arm_dict['name'].split(',')):
        a_dict[name]=a[idx].item()

    # Return arm as in fairseq metrics
    return a_dict

# Process bandit context from fairseq metrics
def bandit_context_from_fairseq(bandit_context_dict, bandit_interaction, pretrain_metrics, finetune_metrics):
    # Context dimensionality
    d_context=len(
                bandit_context_dict['name'].split(',')
            )
    
    # TODO: consider other options
    x=None
    if 'bandit_interaction' in bandit_context_dict['name']:
        # Bandit interactions
        x=int(bandit_interaction)*torch.ones(d_context)
    elif 'training_epoch' in bandit_context_dict['name']:
        # Number of training epoch
        # Assuming loaded metrics contains constant number of epochs per bandit interaction
        x=int(bandit_interaction)*int(pretrain_metrics[-1]['epoch'])*torch.ones(d_context)

    # Return bandit context
    return x

# Get next bandit context
def bandit_context_next(x_prev, bandit_context_dict, pretrain_metrics, finetune_metrics):
    # TODO: consider other options
    if 'bandit_interaction' in bandit_context_dict['name']:
        # Bandit interactions
        x_next=x_prev+1
    elif 'training_epoch' in bandit_context_dict['name']:
        # Number of training epoch
        # Assuming last loaded metrics contains constant number of epochs per bandit interaction
        x_next=x_prev+int(pretrain_metrics[-1]['epoch'])   
    
    # Return next context
    return x_next

# Process bandit reward info from fairseq metrics
def bandit_reward_from_fairseq(bandit_reward_dict, output_dir, pretrain_metrics, finetune_metrics):   
    #####################
    # Validation losses
    #####################
    # Pretraining
    if 'pretrain_val_loss' in bandit_reward_dict:
        # We already have pre-training info available in pretrain_metrics
        if bandit_reward_dict['pretrain_val_loss'] == 'last':
            # Last loss
            y = (- pretrain_metrics[-1]['val_loss'])
        elif bandit_reward_dict['pretrain_val_loss'] == 'delta':
            if len(pretrain_metrics)>1:
                y = (- pretrain_metrics[-1]['val_loss']) - (- pretrain_metrics[-2]['val_loss'])
            else:
                y = float('nan')
    # Finetuning
    elif 'finetune_val_loss' in bandit_reward_dict:
        # We already have pre-training info available in finetune_metrics
        # Process fine-tuned validation loss 
        if bandit_reward_dict['finetune_val_loss'] == 'last':
            # Last loss
            y = (- finetune_metrics[-1]['val_loss'])
        elif bandit_reward_dict['finetune_val_loss'] == 'delta':
            if len(finetune_metrics)>1:
                y = (- finetune_metrics[-1]['val_loss']) - (- finetune_metrics[-2]['val_loss'])
            else:
                y = float('nan')
    # Finetuning
    elif 'finetune_val_accuracy' in bandit_reward_dict:
        # We already have pretraining info available in finetune_metrics
        # Process finetuned validation accuracy 
        if bandit_reward_dict['finetune_val_accuracy'] == 'last':
            # Last accuracy
            y = (finetune_metrics[-1]['val_accuracy'])
        elif bandit_reward_dict['finetune_val_accuracy'] == 'last_normalized':
            # Last normalized accuracy
            y = (finetune_metrics[-1]['val_accuracy'])/100
        elif bandit_reward_dict['finetune_val_accuracy'] == 'delta':
            if len(finetune_metrics)>1:
                y = (finetune_metrics[-1]['val_accuracy']) - (finetune_metrics[-2]['val_accuracy'])
            else:
                y = float('nan')
        elif bandit_reward_dict['finetune_val_accuracy'] == 'delta_normalized':
            if len(finetune_metrics)>1:
                y = ((finetune_metrics[-1]['val_accuracy']) - (finetune_metrics[-2]['val_accuracy']) ) /100
            else:
                y = float('nan')
    #####################
    # Test losses
    #####################
    elif 'test_loss' in bandit_reward_dict:
        # TODO
        pass
    else:
        raise ValueError('Not implemented yet')
    
    # Return computed reward
    return torch.tensor(y)

# Save bandit model posterior
def save_bandit_posterior(bandit, arm_space='None', context=None, posterior_mode='mean_std', sample_shape=torch.Size(), output_dir='./', bandit_interaction=0):
    # Decide on arm space to use
    if arm_space is None:
        arm_space=bandit.arm_space
    
    # And save it for future reference
    if arm_space is not None:    
        # Save arm space
        with gzip.open('{}/model_posterior/arm_space_i{}{}.gz'.format(
                            output_dir,
                            int(bandit_interaction),
                            '' if context is None else '_z{}'.format(context.data.tolist())
                            ), 'wb') as f:
            pickle.dump(arm_space, f)
    
    # Posteriors
    posterior_mean=None
    posterior_std=None
    posterior_samples=None
    # Depending on context
    if context is None:
        if 'mean' in posterior_mode:
            posterior_mean=bandit.reward_model.mean(arm_space).detach()
        if 'std' in posterior_mode:
            posterior_std=bandit.reward_model.standard_deviation(arm_space).detach()     
        if 'sample' in posterior_mode:
            posterior_samples=bandit.reward_model.sample(arm_space, sample_shape).detach()
    else:
        if 'mean' in posterior_mode:
            posterior_mean=bandit.reward_model.mean(context, arm_space).detach()
        if 'std' in posterior_mode:
            posterior_std=bandit.reward_model.standard_deviation(context, arm_space).detach()     
        if 'sample' in posterior_mode:
            posterior_samples=bandit.reward_model.sample(context, arm_space, sample_shape).detach()
    
    if posterior_mean is not None:    
        # Save posterior mean
        with gzip.open('{}/model_posterior/mean_i{}{}.gz'.format(
                            output_dir,
                            int(bandit_interaction),
                            '' if context is None else '_z{}'.format(context.data.tolist())
                            ), 'wb') as f:
            pickle.dump(posterior_mean, f)
        
    if posterior_std is not None:
        # Save posterior std
        with gzip.open('{}/model_posterior/std_i{}{}.gz'.format(
                            output_dir,
                            int(bandit_interaction),
                            '' if context is None else '_z{}'.format(context.data.tolist())
                            ), 'wb') as f:
            pickle.dump(posterior_std, f)
    
    if posterior_samples is not None:
        # Save posterior std
        with gzip.open('{}/model_posterior/samples_i{}{}.gz'.format(
                            output_dir,
                            int(bandit_interaction),
                            '' if context is None else '_z{}'.format(context.data.tolist())
                            ), 'wb') as f:
            pickle.dump(posterior_samples, f)
            
    
