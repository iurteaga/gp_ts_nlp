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

# Update and play bandit, given last bandit interaction information
def update_play_bandit(bandit_config, bandit, x_prev, a_prev, y_prev, pretrain_metrics, finetune_metrics):
    # Contextual bandit
    if 'bandit_context' in bandit_config:
        # Update bandit history
        bandit.update_history(
                observed_context=x_prev,
                played_arm=a_prev,
                observed_reward=y_prev
                )
        
        # Next context
        x_next=bandit_context_next(x_prev, bandit_config['bandit_context'], pretrain_metrics, finetune_metrics)
        
        # Decide next arm
        a_next=bandit.next_action(x_next)
        
    # Non-contextual otherwise
    else:
        x_next=None
        # Update bandit history
        bandit.update_history(
                played_arm=a_prev,
                observed_reward=y_prev
                )
        
        # Decide next arm
        a_next=bandit.next_action()

    # Return next context and action
    return a_next, x_next
    
# Load, update history, decide next arm and save bandit 
def play_bandit(bandit_config_file, bandit_interaction, output_dir, pretrain_metrics, finetune_metrics):
    ### Bandit, based on config parser
    bandit_config = ConfigParser()
    bandit_config.read('{}'.format(bandit_config_file))
    
    # For contextual bandit
    x_prev=None
    if 'bandit_context' in bandit_config:
        # Process bandit context from fairseq information
        x_prev=bandit_context_from_fairseq(bandit_config._sections['bandit_context'], bandit_interaction, pretrain_metrics, finetune_metrics)
    
    # Process bandit arm from fairseq information
    a_prev=bandit_arm_from_fairseq(bandit_config._sections['bandit_arm'], pretrain_metrics, finetune_metrics)
    
    # Process bandit reward from fairseq information
    y_prev=bandit_reward_from_fairseq(bandit_config._sections['bandit_reward'], output_dir, pretrain_metrics, finetune_metrics)
    
    # Debugging:
    print('| BANDIT | i={} | Previous arm: {}'.format(bandit_interaction, a_prev))
    if x_prev is not None:
        print('| BANDIT | i={} | Previous context: {}'.format(bandit_interaction, x_prev))
    print('| BANDIT | i={} | Observed reward: {}'.format(bandit_interaction, y_prev))
    
    # If we have reward for previous arm 
    if not torch.isnan(y_prev):
        # Load bandit object
        bandit_filepath='{}/{}.pt'.format(
                output_dir,
                os.path.basename(bandit_config_file) # Filename
                )
        with gzip.open(bandit_filepath, 'rb') as f:
            # Use torch and dill for gpytorch objects
            bandit=torch.load(f, pickle_module=dill)
            
            # Update and play bandit
            a_next, x_next=update_play_bandit(bandit_config, bandit, x_prev, a_prev, y_prev, pretrain_metrics, finetune_metrics)
            
            # Debug bandit posterior
            save_bandit_posterior(
                                    bandit,
                                    arm_space=None,
                                    context=x_next,
                                    posterior_mode='mean_std',
                                    sample_shape=None,
                                    output_dir=output_dir,
                                    bandit_interaction=bandit_interaction
                                )
            
            # If we want to plot
            '''
            plot_bandit_model_posterior(
                            bandit,
                            arm_space=None, # Default arm space
                            context=x_next, # For this specific context # TODO: for all possible contexts?
                            plot_mode='mean_std',
                            sample_shape=None,
                            plot_filename='{}/model_posterior/posterior_i{}{}.pdf'.format(
                                output_dir,
                                int(bandit_interaction),
                                '' if x_next is None else '_z{}'.format(x_next.data.tolist())
                                )
                            )
            '''
            
            # Save bandit object
            with gzip.open(bandit_filepath, 'wb') as f:
                # Use torch and dill for gpytorch objects
                torch.save(bandit, f, pickle_module=dill)
    else:
        # Repeat with last arm
        a_next=a_prev
        
    # return arm to play
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
    # Run bandit
    # TODO: which finetune_metrics to use for the bandit?
    # TODO: Call script with only one for now!!!
    a_t=play_bandit(path_to_bandit_config, bandit_interaction, output_dir, pretrain_metrics, finetune_metrics)
    
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
