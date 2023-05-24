#!/usr/bin/python

# Imports: python modules
import sys, os, re, time
import argparse
from configparser import ConfigParser
import yaml
import gzip, pickle, dill
import copy

# Import fairseq checkpoint utilities
from fairseq import checkpoint_utils

# Imports: our own modules
# Add path
sys.path.append('./bandits')
# GP models
from gp_models import *
# Bandit modules
from bandits import *
from bandit_reward_models import * 
from bandit_plotting import * 

# Checkpoint and metric utils
from cpkt_metrics_utils import *
# Bandit utils
from bandit_utils import *
# Overall plotting
from plotting_functions import *

# Random arm value from params dictionary 
def random_arm_pull(bandit_config_file):
    ### Bandit arm information, based on config parser
    bandit_config = ConfigParser()
    bandit_config.read('{}'.format(bandit_config_file))
    bandit_arm_dict=bandit_config._sections['bandit_arm']
    
    # Dimensionality of arms
    d_arms=len(
            bandit_arm_dict['name'].split(',')
        )

    # Points within arms
    arm_range=bandit_arm_dict['range'].split(',')

    if 'delta_per_arm' in bandit_arm_dict:
        # Range based on delta
        a_points=torch.arange(
                    float(arm_range[0]),
                    float(arm_range[1]),
                    float(bandit_arm_dict['delta_per_arm'])
                ) 
    elif 'n_points_per_arm' in bandit_arm_dict:
        # Equally spaced based on n_points
        a_points=torch.linspace(
                    float(arm_range[0]),
                    float(arm_range[1]),
                    int(bandit_arm_dict['n_points_per_arm'])+1
                )
    else:
        raise ValueError('Need to specify delta or number of points per arm')
    
    # Draw uniformaly per-arm, within range
    a_next=torch.zeros(d_arms)
    for d_arm in torch.arange(d_arms):
        a_next[d_arm]=a_points[torch.randperm(a_points.size()[0])][0]

    # Return arm to play
    return a_next

def update_fairseq_from_bandit_arm(a_t, bandit_config_file, path_to_pretrain_config, path_to_finetune_config):    
    ### Bandit, based on config parser
    bandit_config = ConfigParser()
    bandit_config.read('{}'.format(bandit_config_file))
    
    # Bandit arm definition section
    bandit_arm_config=bandit_config._sections['bandit_arm']
    
    # Process bandit arm to fairseq information dictionary
    a_dict=bandit_arm_to_fairseq(bandit_arm_config, a_t)
    
    # Pretraining
    if bandit_arm_config['scope']=='pretraining':
        print('Updating fairseq pretraining config file')
        update_fairseq_config_file(path_to_pretrain_config, a_dict)
    # Finetuning
    if bandit_arm_config['scope']=='finetuning':
        print('Updating fairseq finetuning config file')
        update_fairseq_config_file(path_to_finetune_config, a_dict)

def update_fairseq_config_file(config_filepath, params_to_update):
    # Load from fairseq config file
    with open(config_filepath, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    # Update in dict
    # For masked_lm parameters
    if config_dict['task']['_name']=='masked_lm':
        for param_to_update,param_value in params_to_update.items():
            config_dict['task'][param_to_update]=param_value
    else:
        raise ValueError('Unclear on what fairseq config item {} bandit_arm parameters {} correspond to'.format(config_dict, params_to_update))
    
    # Save to fairseq config file
    # Note this changes yaml format, does fairseq still work with this?
    with open(config_filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=None)
        
def main(path_to_bandit_config, bandit_interaction, output_dir, path_to_pretrain_config, path_to_finetune_config_dir):
    #########################################
    # Process checkpoint and metrics
    # Pretraining
    print('Pretraining output dir = {}'.format(output_dir))
    pretrain_metrics=process_ckpt_metrics(output_dir, bandit_interaction, ckpt_info='Pretraining')
    # Finetuning
    if path_to_finetune_config_dir is not None:
        print('Finetuning config dir = {}'.format(path_to_finetune_config_dir))
        # For all finetuned models
        finetuning_dirs=os.listdir(path_to_finetune_config_dir)
        for this_finetuning in np.sort(finetuning_dirs):
            print('Finetuning output dir = {}/{}'.format(path_to_finetune_config_dir, this_finetuning))
            finetune_metrics=process_ckpt_metrics('{}/{}'.format(path_to_finetune_config_dir, this_finetuning), bandit_interaction, ckpt_info='Finetuning')
    else:
        finetune_metrics=None
    
    #########################################
    # Run random arm
    a_t=random_arm_pull(path_to_bandit_config)
    
    #########################################
    # Update Fairseq config file, if needed
    update_fairseq_from_bandit_arm(a_t, path_to_bandit_config, path_to_pretrain_config, path_to_finetune_config_dir)
    
#########################################  
# Main program is not executed when the module is imported  
if __name__ == '__main__':
    # Load arguments
    parser = argparse.ArgumentParser(
            description='Script to load Fairseq checkpoint and update config string'
        )
    parser.add_argument(
        '-path_to_bandit_config',
        type=str,
        default='./bandit_config/gp_bandit_config.ini',
        help='Filepath to Bandit object file'
        )
    parser.add_argument(
        '-bandit_interaction',
        type=int,
        default='1',
        help='Bandit interaction'
        )
    parser.add_argument(
        '-path_to_output_dir',
        type=str,
        default='./nlp_bandit_experiments',
        help='Directory for output content'
        )
    parser.add_argument(
        '-path_to_pretrain_config',
        type=str,
        default='./fairseq_config/bandit_pretrain_params',
        help='Filepath to Fairseq pretraining config file'
        )
    parser.add_argument(
        '-path_to_finetune_config_dir',
        type=str,
        default=None,
        help='Filepath to Fairseq finetuning config directory'
        )
    # Get arguments from script
    args = parser.parse_args()
    
    # If empty
    if args.path_to_finetune_config_dir == 'None':
        path_to_finetune_config_dir=None
    else:
        path_to_finetune_config_dir=str(args.path_to_finetune_config_dir)
    
    # Call main
    main(
        args.path_to_bandit_config,
        args.bandit_interaction,
        args.path_to_output_dir,
        args.path_to_pretrain_config,
        path_to_finetune_config_dir,
    )
