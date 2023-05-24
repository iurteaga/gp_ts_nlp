#!/usr/bin/python

# General imports
import sys,os,time
# Glob
import glob
# Datetime
from datetime import datetime
# Pickles and compressed
import gzip, pickle
# Science
import numpy as np
# Plotting
import matplotlib.pyplot as plt
from matplotlib import colors

import os

# Colors
my_colors=[
    colors.cnames['black'],   
    colors.cnames['grey'],
    colors.cnames['orange'],
    colors.cnames['red'],
    colors.cnames['tomato'],
    colors.cnames['skyblue'],
    colors.cnames['cyan'],
    colors.cnames['aqua'],
    colors.cnames['blue'],
    colors.cnames['palegreen'],
    colors.cnames['lime'],
    colors.cnames['green'],
    colors.cnames['yellow'],
    colors.cnames['purple'],
    colors.cnames['fuchsia'],
    colors.cnames['pink'],
    colors.cnames['peru'],
    colors.cnames['saddlebrown'],
    colors.cnames['chocolate'],
    colors.cnames['burlywood']
]

# Print stdout and stderrs
def print_out_file(experiment_stdout_file):
    # Stdout
    with open('{}'.format(experiment_stdout_file), 'rb') as f:
        [print(line) for line in f.read().splitlines()]

# Print stdout and stderrs
def print_filtered_out_file(experiment_stdout_file, filter_str='train'):
    # Stdout
    with open('{}'.format(experiment_stdout_file), 'rb') as f:
        for line in f.read().splitlines():
            if '| {} |'.format(filter_str) in str(line):
                print(line)
                
# TODO: more detailed processing: different train/val losses depending on task!
def get_loss_from_out_file(experiment_stdout_file, loss_type='train', max_interactions=500, plot_save=None):
    # For all interactions and experiments    
    loss=np.zeros(max_interactions)
              
    with open('{}'.format(experiment_stdout_file), 'rb') as f:
        for line in f.read().splitlines():
            if '| {} |'.format(loss_type) in str(line):
                info=str(line).split(' |')
                #print(info)
                epoch=None
                this_loss=None
                # Process information
                for item in info:
                    if ' epoch ' in item:
                        epoch=int(item.split('epoch ')[1])
                    if ' loss ' in item:
                        this_loss=float(item.split('loss ')[1])
                # Print and save information
                #print('Epoch {} with loss {}'.format(epoch,val_loss))
                loss[experiment_id,epoch-1]=this_loss
                
    return loss

def plot_loss_from_loss_array(experiments_loss, loss_type='train', plot_save=None):
    n_experiments, max_interactions=experiments_loss.shape
    for experiment_id in np.arange(n_experiments):
        # Plot
        plt.plot(
            np.arange(max_interactions),
            experiments_loss[experiment_id,:],
            my_colors[experiment_id],
            label='{}'.format(
                experiment['job_name']
            )
        )
        plt.xlabel('{} epoch'.format(loss_type))
        plt.ylabel('{} loss'.format(loss_type))
        plt.title('{} loss over epochs'.format(loss_type))
        plt.autoscale(enable=True, tight=True)
        legend = plt.legend(
            bbox_to_anchor=(1.05,1.05),
            loc='upper left',
            ncol=1,
            shadow=True
        )

    # Whether to show or save the plot
    if plot_save is None: 
        plt.show()
    else:
        plt.savefig('{}_{}.pdf'.format(plot_save, loss_type), format='pdf', bbox_inches='tight')
        plt.close()

# Plot metrics pickle
# metrics_path should always be path to a valid pickle file containing metrics
# plot_dir should be the path of where the plot files should be saved
# When plot_filename is None, display the plot directly
def plot_metrics_from_pickle(metrics_path, plot_dir=None):
    with gzip.open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    
    plot_metrics(metrics, plot_dir)

# Plot metrics
# metrics: a list of dictionaries. Each dictionary should have the following keys:
# 'epoch', 'iterations_in_epoch', 'train_time', 'train_loss', 'val_loss', 'mask_prob', 'leave_unmasked_prob', 'random_token_prob'
# plot_dir: the directory to save pdf files of plots
# When plot_dir is None, display plots directly.
def plot_metrics(metrics, ckpt_info='Pre-training', plot_dir=None):

    # TODO: revise/improve this
    if ckpt_info=='Pre-training':
        items = ['train_loss', 'val_loss', 'mask_prob', 'leave_unmasked_prob', 'random_token_prob']
    elif ckpt_info=='Fine-tuning':
        items = ['train_loss', 'train_nll_loss', 'val_loss', 'val_nll_loss']
        if 'train_accuracy' in metrics[-1]:
            items+=['train_accuracy']
        if 'val_accuracy' in metrics[-1]:
            items+=['val_accuracy']
    else:
        items = ['train_loss', 'val_loss']
    
    for item in items:
        fig, ax = plt.subplots()
        ax.plot([entry['epoch'] for entry in metrics], [entry[item] for entry in metrics], marker='o')
        ax.set_title('{} for each epoch'.format(item))
        ax.set_xticks([entry['epoch'] for entry in metrics])
        plt.autoscale(enable=True, tight=True)
        if plot_dir == None:
            plt.show()
        else:
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig("{}/{}.pdf".format(plot_dir,item), format='pdf', bbox_inches='tight')
            plt.close()
    

