#!/usr/bin/python
"""
@authors: Iñigo Urteaga and Moulay-Zaidane Draidia
"""

# Imports: python modules
import sys, os, re, time
import argparse

# Load Fairseq config from file to string
def load_fairseq_config_string(config_filepath):
    config_data=None
    # Try opening the config file
    with open('{}'.format(config_filepath), 'r') as config:
        config_data=config.read()
        # Remove double newlines
        config_data=re.sub('\n\n','\n', config_data)
        # Ignore commented lines
        config_data=re.sub('#.*\n','', config_data)
        # Put all together with spaces
        config_data=re.sub('\n',' ', config_data)
        
    # Return content
    return config_data

# Update Fairseq config string with new parameter values
def update_fairseq_config_string(config_string, params_to_update):
    # Update parameter values
    for param_to_update,param_value in params_to_update.items():
        config_string=re.sub('--{} (.*?) '.format(param_to_update), '--{} {} '.format(param_to_update,param_value), config_string)
    
    # Return modified config string
    return config_string

def update_fairseq_config_file(config_filepath, params_to_update):
    # Try opening the config file
    with open('{}'.format(config_filepath), 'r+') as config:
        # Read old config
        new_config_data=config.read()
        # Update parameter values
        for param_to_update,param_value in params_to_update.items():
            new_config_data=re.sub('--{} (.*?)\\n'.format(param_to_update), '--{} {}\\n'.format(param_to_update,param_value), new_config_data)
        
        # Write to file and close
        config.seek(0)
        config.write(new_config_data)
        config.truncate()
        config.close()

def main(config_filepath, save_dir, restore_file):
    # Call function
    config_data=load_fairseq_config_string(config_filepath)

    # Replace save_dir, if given
    if save_dir is not None:
        # Replace save dir
        config_data=re.sub('--save-dir (.*?) ', '--save-dir {} '.format(save_dir), config_data)

    # Replace restore_file, if given
    if restore_file is not None:
        # Replace save dir
        config_data=re.sub('--restore-file (.*?) ', '--restore-file {} '.format(restore_file), config_data)

    # Print content, to use within shell script
    print(config_data)

#########################################  
# Main program is not executed when the module is imported
if __name__ == '__main__':    
    parser = argparse.ArgumentParser(
            description='Script to load Fairseq config from file to string'
        )
    parser.add_argument(
        '-fairseq_config_filepath',
        type=str,
        default='./fairseq_config/pretrain_params_string',
        help='Filepath to Fairseq config file'
        )
    parser.add_argument(
        '-save_dir',
        type=str,
        default='./nlp_bandit_experiments',
        help='Directory for output content'
        )
    
    parser.add_argument(
        '-restore_file',
        type=str,
        default='None',
        help='Path to model checkpoint to restore training from'
        )
    
    # Get arguments from script
    args = parser.parse_args()
    
    # Call main
    main(
        args.fairseq_config_filepath,
        args.save_dir,
        args.restore_file
    )
