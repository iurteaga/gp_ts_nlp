#!/usr/bin/python
"""
@authors: Iñigo Urteaga and Moulay-Zaidane Draidia
"""

# Imports: python modules
import sys, os, re, time
import argparse
from configparser import ConfigParser
import gzip, pickle, dill
import copy

# Imports: our own modules
# Add path
sys.path.append('./bandits')
# Aux functions
from aux_functions import *
# GP models
from gp_models import *
# Bandit modules
from bandits import *
from bandit_reward_models import * 
from bandit_plotting import * 

# Other Bandit utils
from bandit_utils import *

# Initialize bandit's arm space from params dictionary 
def init_bandit_arm_space(bandit_arm_dict):   
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
    
    # Make meshgrid
    per_arm_meshgrid=torch.meshgrid([a_points]*d_arms)
    arm_space=torch.stack(
            per_arm_meshgrid,
            axis=-1
        ).reshape(-1, d_arms) # n_points in dim==1
    
    return arm_space
    
# Initialize and save GP based bandit 
def main(bandit_config_file, output_dir):    
    
    # Bandit-file path
    bandit_filepath='{}/{}.pt'.format(
        output_dir,
        os.path.basename(bandit_config_file) # Filename
    )
                    
    # If bandit object already exists
    # e.g., because we are continuing from a previous training
    if os.path.exists(bandit_filepath):
        # Make sure it can be loaded
        with gzip.open(bandit_filepath, 'rb') as f:
            # Use torch and dill for gpytorch objects
            bandit=torch.load(f, pickle_module=dill)
            print('Bandit object {} already exists and is ready!'.format(bandit_filepath))
        
        # Also make sure model_posterior is ready
        if ~ os.path.exists('{}/model_posterior'.format(output_dir)):
            # Debug bandit prior
            os.makedirs('{}/model_posterior'.format(
                            output_dir
                            ),
                            exist_ok=True
            )
        
    # Otherwise, initialize bandit object based on config
    else:    
        ### Bandit, based on config parser
        bandit_config = ConfigParser()
        bandit_config.read('{}'.format(bandit_config_file))
        
        # GP training options
        gp_training_options={
                'loss':eval(
                        bandit_config.get(
                                'training',
                                'loss',
                                fallback='gpytorch.mlls.ExactMarginalLogLikelihood()')
                           ),
                'n_train_max_iters':bandit_config.getint(
                                'training',
                                'n_train_max_iters',
                                fallback=100),
                'loss_epsilon':bandit_config.getfloat(
                                'training',
                                'loss_epsilon',
                                fallback=0.01),
                'optimizer':bandit_config.get(
                                'training',
                                'optimizer',
                                fallback=torch.optim.Adam),
                'optimizer_params':cast_dict_values(
                                    bandit_config._sections['optimization_params'],
                                    float
                                    ),
            }

        # Create arm_space
        arm_space=init_bandit_arm_space(
                    bandit_config._sections['bandit_arm']
                    )
        
        # Bandit algorithm
        bandit_algorithm=bandit_config._sections['bandit_algorithm']
        
        # Create the GP-based bandit
        bandit=None
        # Contextual bandit
        if 'bandit_context' in bandit_config:
            # Dimensionality of arms
            d_context=len(
                    bandit_config['bandit_context']['name'].split(',')
                )
            this_x=torch.ones(d_context)
            
            # GP contextual model definition (without training input/output yet)
            gp_model=ExactContextualGPModel(
                            gp_input=None,
                            y=None,
                            d_context=d_context,
                            mean_functions=eval(bandit_config.get('gp_model', 'mean_functions')),
                            kernel_functions=eval(bandit_config.get('gp_model', 'kernel_functions')),
                            action_context_composition=eval(bandit_config.get('gp_model', 'action_context_composition')),
                            likelihood=eval(bandit_config.get('gp_model', 'llh_function')),
                        )
            
            # Initialize if needed
            if 'gp_model_initialization' in bandit_config:
                # Load and cast init values
                model_init_dir=cast_dict_values(
                                cast_dict_values(
                                    bandit_config._sections['gp_model_initialization'],
                                    float
                                ),
                                torch.tensor
                            )
                # Initialize with provided values
                # Need to hack, due to how we define mean/covar modules
                for name, val in model_init_dir.items():
                    module, var = name.split('.',1)
                    eval('gp_model.{}'.format(module)).initialize(**{var:val})
            
            # Bandit reward model
            bandit_reward_model=GPContextualRewardModel(
                        gp_model=gp_model,
                        likelihood_model=eval(
                                        bandit_config.get('gp_model', 'llh_function')
                                        ),
                        gp_training=gp_training_options,
                    )
            
            # Instantiate and return bandit class
            bandit=ContinuousArmContextualBandit(
                        d_context=d_context,
                        arm_space=arm_space,
                        reward_model=bandit_reward_model,
                        algorithm=bandit_algorithm
                    )
        
        # Non-contextual otherwise
        else:
            this_x=None
            # GP model definition (without training input/output yet)
            gp_model=ExactGPModel(
                            a=None,
                            y=None,
                            mean_function=eval(bandit_config.get('gp_model', 'mean_function')),
                            kernel_function=eval(bandit_config.get('gp_model', 'kernel_function')),
                            likelihood=eval(bandit_config.get('gp_model', 'llh_function')),
                        )

            # Initialize if needed
            if 'gp_model_initialization' in bandit_config:
                # Load and cast init values
                model_init_dir=cast_dict_values(
                                cast_dict_values(
                                    bandit_config._sections['gp_model_initialization'],
                                    float
                                ),
                                torch.tensor
                            )
                # Initialize with provided values
                gp_model.initialize(**model_init_dir)
            
            # Bandit reward model
            bandit_reward_model=GPRewardModel(
                        gp_model=gp_model,
                        likelihood_model=eval(
                                            bandit_config.get('gp_model', 'llh_function')
                                            ),
                        gp_training=gp_training_options,
                    )
            
            
            # Instantiate and return bandit class
            bandit=ContinuousArmBandit(
                        arm_space=arm_space,
                        reward_model=bandit_reward_model,
                        algorithm=bandit_algorithm
                    )
        
        # Debug bandit prior
        os.makedirs('{}/model_posterior'.format(
                            output_dir
                            ),
                            exist_ok=True
            )
        
        # Save posterior
        save_bandit_posterior(
                            bandit,
                            arm_space=None,
                            context=this_x,
                            posterior_mode='mean_std',
                            sample_shape=None,
                            output_dir=output_dir,
                            bandit_interaction=0
                        )
        # Plot
        '''
        plot_bandit_model_posterior(
                        bandit,
                        arm_space=None, # Default arm space
                        context=this_x, # For no specific context # TODO: for all possible contexts?
                        plot_mode='mean_std',
                        sample_shape=None,
                        plot_filename='{}/model_posterior/posterior_i0.pdf'.format(
                            output_dir,
                            '' if this_x is None else '_z{}'.format(this_x.data.tolist())
                            )
                        )
        '''
        # Save bandit object
        os.makedirs(output_dir, exist_ok=True)
        with gzip.open(bandit_filepath, 'wb') as f:
            # Use torch and dill for gpytorch objects
            torch.save(bandit, f, pickle_module=dill)    
            print('Bandit object {} created!'.format(bandit_filepath))
            
if __name__ == '__main__':
    # Load arguments
    parser = argparse.ArgumentParser(
            description='Script to initialize ans save a bandit model'
        )
    parser.add_argument(
        '-bandit_config',
        type=str,
        default='./bandit_config/gp_bandit_config.ini',
        help='Filepath to Bandit object file'
        )
    parser.add_argument(
        '-output_dir',
        type=str,
        default='./nlp_bandit_experiments',
        help='Directory for output content'
        )
        
    # Get arguments from script
    args = parser.parse_args()
    
    # Call main
    main(
        args.bandit_config,
        args.output_dir
    )
