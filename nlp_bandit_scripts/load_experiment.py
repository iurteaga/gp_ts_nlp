"""
@authors: Iñigo Urteaga and Moulay-Zaidane Draidia
"""

import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import gzip
import pickle

import dill
import torch

from os import listdir
from os.path import isfile, join
import os

import sys
sys.path.append('../bandits')
# GP models
from gp_models import *
# Bandit modules
from bandits import *
from bandit_reward_models import *
from bandit_plotting import *



class Experiment:

    def __init__(self, out_path, bdt_path=None):

        # paths
        self.out_path = out_path
        self.bdt_path = bdt_path

        try:
            self.results = self.load_out()
        except FileNotFoundError:
            print("File {} was not found".format(self.out_path))



        if self.bdt_path:
            try:
                self.load_bandit()
            except FileNotFoundError:
                print("File {} was not found".format(self.bdt_path))


    def load_out(self):

    # For all interactions and experiments

        
        state = None
        interaction = 0
        
        meta_dict = {interaction: {'pretrain': {'train':{}, 'valid':{}}, 'finetune': {'train':{}, 'valid':{}}}}
        meta_dict['evaluation'] = {}


        with open('{}'.format(self.out_path), 'rb') as f:

            for l in f.read().splitlines():

                line = str(l)

                # identifying the start of a phase

                # TODO: more robust implementation

                # if 'Starting interaction' in line:
                if 'Starting' in line and ' interaction ' in line:
                    if meta_dict[interaction] != {'pretrain': {'train':{}, 'valid':{}}, 'finetune': {'train':{}, 'valid':{}}}:
                        interaction += 1

                    meta_dict[interaction] = {'pretrain': {'train':{}, 'valid':{}}, 'finetune': {'train':{}, 'valid':{}}}

                if 'Starting pre-training' in line:
                    state = 'pretrain'

                if 'Starting fine-tuning' in line:
                    state = 'finetune'
                    # TODO: revise this to filter only glue_ or ebay_
                    #ftask = str(line).split('/')[-1]#.split('_')[-1][:-1]
                    ftask = line.strip("'").split('/')[-1]
                    for loss_type in ['train', 'valid']:
                        meta_dict[interaction]['finetune'][loss_type][ftask] = {}

                if 'Starting evaluating' in line:
                    state = 'evaluation'
                    if 'evaluation' not in meta_dict[interaction].keys():
                        meta_dict[interaction]['evaluation'] = {}

                if 'Done script' in line:
                    state = 'done'



                # identifying the end of a phase

                for end_phase in ['Finished {}'.format(phase) for phase in ['fine-tuning', 'pre-training', 'evaluating']]:
                    if end_phase in line:
                        state = None


                # update epoch level information

                for loss_type in ["train", "valid"]:
                    # if loss_type == "valid":
                    #     loss_tag = "valid on 'valid' subset"
                    #     if '| {} |'.format(loss_tag) in str(line):
                    #         info = str(line).split('|')
                    #         info_dict = echolist_to_dict(info)
                    #
                    # elif loss_type == "train":
                    loss_tag = f"[{loss_type}][INFO] -"
                    if loss_tag in str(line):
                        info = str(line).split('|')
                        info_dict = echolist_to_dict(info)
                        if state == 'pretrain':
                            meta_dict[interaction]['pretrain'][loss_type][info_dict['epoch']] = info_dict

                        elif state == 'finetune':
                            meta_dict[interaction]['finetune'][loss_type][ftask][info_dict['epoch']] = info_dict

                # evaluation metrics

                if '| Evaluation |' in line:
                    tsk = line.split('Task=')[1].split(' ')[0]
                    met = line.split('Metric=')[1].split(' ')[0]
                    val = float(line.split('|')[-2].strip())

                    if tsk in meta_dict[interaction]['evaluation'].keys():
                        meta_dict[interaction]['evaluation'][tsk][met] = val 
                    else:
                        meta_dict[interaction]['evaluation'][tsk] = {met: val}

                # if '| Evaluation |' in line:
                #     tsk = line.split('Task=')[1].split(' ')[0]
                #     met = line.split('Metric=')[1].split(' ')[0]
                #     val = float(line.split('|')[-2].strip())
                #     if tsk in meta_dict['evaluation'].keys():
                #         meta_dict['evaluation'][tsk][met] = val 
                #     else:
                #         meta_dict['evaluation'][tsk] = {met: val}

        self.complete = state == 'done'
        if not self.complete:
            print('Incomplete experiment: {} \n'.format(self.out_path))

        return meta_dict


    def load_bandit(self):
        self.load_bandit_obj()
        self.load_posteriors()

    def load_bandit_obj(self):
        with gzip.open(self.bdt_path, 'rb') as f:
            # Use torch and dill for gpytorch objects
            bandit = torch.load(f, pickle_module=dill)

        self.bandit = bandit
        
        # Updated by iurteaga
        obs_reward = float('nan')*torch.ones(bandit.t)
        played_arm = float('nan')*torch.ones(bandit.t) # TODO: This only works for 1-dimensional arms!
        
        obs_reward[:bandit.t] = bandit.observed_rewards[:bandit.t]
        played_arm[:bandit.t] = bandit.played_arms[:bandit.t,0] # TODO: This only works for 1-dimensional arms!

        self.bandit_obj = {}

        for inter_i in range(bandit.t):
            self.bandit_obj[inter_i] = {'reward': obs_reward[inter_i], 'action': played_arm[inter_i]}

    def load_posteriors(self):
    
        model_posterior = {}
        
        mp_path = '{}model_posterior/'.format(self.bdt_path.rstrip(self.bdt_path.split('/')[-1]))
        mp_files = [f for f in listdir(mp_path) if isfile(join(mp_path, f))]
        
        for file in mp_files:
            try:
                interaction = int(file.rstrip('.gz').split('i')[-1])
                if interaction not in model_posterior.keys():
                    model_posterior[interaction] = {}

                with gzip.open("{}{}".format(mp_path, file), 'rb') as gz:
                    mp_x_i = pickle.load(gz)

                if 'mean' in file:
                    model_posterior[interaction]['mean'] = mp_x_i
                elif 'arm' in file:
                    model_posterior[interaction]['arm'] = mp_x_i
                elif 'std' in file:
                    model_posterior[interaction]['std'] = mp_x_i
            except:
                print("File {} could not be read".format(file))
                
        self.mod_post = dict(sorted(model_posterior.items()))

    def get_pretrain_m(self, m, t_or_v='valid', which_epoch='all', verbose=True):
        '''
        which_epoch: if [str] takes in either 'all' or  'last' 
            indicate whether all the data for the epoch should be collected or just the value for the last epoch
            if [int], then which epoch to get

        m: [str] one of the pretraining metrics
        '''

        num_epoch = len(self.results[0]['pretrain'][t_or_v])
        num_inter = len([i for i in list(self.results.keys()) if i != 'evaluation'])


        # X axis
        x_count = 0
        
        # Y axis, metric value
        metric = []
        for i in range(num_inter):
            try:
                if which_epoch == 'all':
                    for e in range(1, num_epoch + 1):
                        metric += [self.results[i]['pretrain'][t_or_v][e][m]]
                        x_count += 1
                elif which_epoch == 'last':
                    last_epoch = list(self.results[i]['pretrain'][t_or_v].keys())[-1]
                    metric += [self.results[i]['pretrain'][t_or_v][last_epoch][m]]
                    x_count += 1
                else :
                    this_epoch = list(self.results[i]['pretrain'][t_or_v].keys())[which_epoch]
                    metric += [self.results[i]['pretrain'][t_or_v][this_epoch][m]]
                    x_count += 1
            except:
                if verbose :
                    if self.complete == False:
                        print('issue with retrieving finetune metric from incomplete experiment {}'.format(self.out_path))
                    else:
                        print('issue with retrieving finetune metric from experiment {}'.format(self.out_path))

        x_axis = [x for x in range(x_count)] 

        return {'x': x_axis, 'y': metric}


    def get_finetune_m(self, task, m, t_or_v='valid', which_epoch='all', verbose=True):
        '''
        which_epoch: if [str] takes in either 'all' or  'last' 
            indicate whether all the data for the epoch should be collected or just the value for the last epoch
            if [int], then which epoch to get

        metric: [str] one of the pretraining metrics
        '''

        num_epoch = len(self.results[0]['finetune'][t_or_v][task])
        num_inter = len([i for i in list(self.results.keys()) if i != 'evaluation'])

        # X axis
        x_count = 0
        
        # Y axis, metric value
        metric = []
        for i in range(num_inter):
            try:
                if which_epoch == 'all':
                    for e in range(1, num_epoch + 1):
                        metric += [self.results[i]['finetune'][t_or_v][task][e][m]]
                        x_count += 1
                elif which_epoch == 'last':
                    last_epoch = list(self.results[i]['finetune'][t_or_v][task].keys())[-1]
                    metric += [self.results[i]['finetune'][t_or_v][task][last_epoch][m]]
                    x_count += 1
                else :
                    this_epoch = list(self.results[i]['finetune'][t_or_v][task].keys())[which_epoch]
                    metric += [self.results[i]['finetune'][t_or_v][task][this_epoch][m]]
                    x_count += 1
            except Exception as e:
                if verbose:
                    if task == 'sts_b' and m == 'accuracy':
                        print('Sts-b is not validated with accuracy')
                    # elif self.complete == False:
                    #     print('issue with retrieving finetune metric from incomplete experiment {}'.format(self.out_path))
                    else:
                        # TODO: understand why these errors occur
                        print('issue with retrieving finetune metric from experiment {} at interaction {}'.format(self.out_path, i))
                        print(e)
                

        x_axis = [x for x in range(x_count)] 

        return {'x': x_axis, 'y': metric}

    def get_eval_m(self, task, m='accuracy'):

        num_inter = len([i for i in list(self.results.keys())])

        # X axis
        x_count = 0
        
        # Y axis, metric value
        metric = []
        for i in range(num_inter-1):
            try:
                metric += [self.results[i]['evaluation'][task][m]]
                x_count += 1
            except Exception as e:
                if self.complete == False:
                    print('issue with retrieving evaluation metric from incomplete experiment {}'.format(self.out_path))
                else:
                    # TODO: understand why these errors occur
                    print('issue with retrieving evaluation metric from experiment {}'.format(self.out_path))
                    print(e)

        x_axis = [x for x in range(x_count)] 

        return {'x': x_axis, 'y': metric}


    def get_mod_post(self, iteraction):
        xy_dict = {
            'x':self.mod_post[iteraction]['arm'].flatten(),
            'y':self.mod_post[iteraction]['mean'].flatten(),
            'error':self.mod_post[iteraction]['std'].flatten()
        }
        return xy_dict

    def get_actions(self):

        mask_prob_results = {'x':[], 'y':[]}
        for i in self.bandit_obj.keys():
            mask_prob_results['x'] += [i]
            mask_prob_results['y'] += [float(self.bandit_obj[i]['action'])]

        return mask_prob_results

    def get_rewards(self):

        reward_results = {'x':[], 'y':[]}
        for i in self.bandit_obj.keys():
            reward_results['x'] += [i]
            reward_results['y'] += [float(self.bandit_obj[i]['reward'])]

        return reward_results

    def view_mod_post(self):

        # Create figure
        fig = go.Figure()

        num_interactions = [i for i in self.mod_post.keys()]

        # Add traces, one for each slider step
        for interaction in num_interactions:
            modpost_i = self.get_mod_post(interaction)
            fig.add_trace(
                go.Scatter(
                    visible=False,
                    line=dict(color="#00CED1", width=6),
                    name="𝜈 = " + str(interaction),
                    x= modpost_i['x'],
                    y= modpost_i['y'],
                    error_y=dict(type='data', array=modpost_i['error'],visible=True)
                ))

        # Make 10th trace visible
        fig.data[-10].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": "Slider switched to interaction: " + str(i)}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=len(fig.data) - 10,
            currentvalue={"prefix": "Interaction: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders
        )

        fig.show()

###################
#### CONFIDENCE ###
###################

def get_conf_finetune(exp, task, metric, loss_type):
    '''
    exp: list of experiment objects
    '''
    
    meta_y = {}
    for i in range(len(exp)):
        exp_i = exp[i]
        Y_seed_i = exp_i.get_finetune_m(task, metric, loss_type)['y']
        if Y_seed_i != []:
            X = exp_i.get_finetune_m(task, metric, loss_type)['x']
            meta_y[i] = Y_seed_i

    standard_deviations = []
    for j in range(len(meta_y[0])):
        values_inter_i = []
        for i in range(len(meta_y)):
            values_inter_i += [meta_y[i][j]]
        standard_deviations += [np.std(values_inter_i)]

    averages = []
    for j in range(len(meta_y[0])):
        values_inter_i = []
        for i in range(len(meta_y)):
            values_inter_i += [meta_y[i][j]]
        averages += [np.mean(values_inter_i)]

    return {'x': X, 'y':averages, 'error':standard_deviations}

def get_conf_pretrain(exp, metric, loss_type):
    '''
    exp: list of experiment objects
    '''
    
    meta_y = []
    for exp_i in exp:
        Y_seed_i = exp_i.get_pretrain_m(metric, loss_type)['y']
        if Y_seed_i != []:
            X = exp_i.get_pretrain_m(metric, loss_type)['x']
            meta_y += [Y_seed_i]
            
    return {'x': X, 'y':np.mean(meta_y, axis=0), 'error':np.std(meta_y, axis=0)}



###################
##### HELPER ######
###################

def load_pickle_metric(m_path):
    with gzip.open(m_path + 'metrics.picklegz', 'rb') as f:
        metrics = pickle.load(f)
    return metrics

def echolist_to_dict(echo_list):
    echo_dict = {}
    for item in echo_list:
        try:
            if item.lstrip().rstrip().count(' ') == 1:
                params = item.lstrip().rstrip().split(' ')
                if len(params) == 2:
                    echo_dict[params[0]] = float(params[1])
            elif "epoch" in item:
                echo_dict["epoch"] = int(item.split('epoch ')[1].strip(' '))
        except:
            pass
    return echo_dict

###################
## LOAD EXP DICT ##
###################

def get_nondir_files(path):
    return [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)) and file != '.DS_Store']

def process_experiments(experiments, base_dir='./nlp_bandit_experiments'):
    for experiment in experiments:
        # Classic
        # set seed to for experiements where no seed in specified
        seed = 1
        if 'classic' in experiment['training_script']:
            finetune_config=''
            for param in experiment['training_script_params']:
                if '-p' in param:
                    pretrain_config=param.split('-p ')[1].split('fairseq_config/krylov/')[1]
                elif '-f' in param:
                    finetune_config=param.split('-f ')[1].split('fairseq_config/krylov/')[1]
                elif '-s' in param:
                    seed=param.split('-s ')[1]
            
            if finetune_config == '':
                # Stdout filename
                out_file='{}/{}/seed_{}/{}_seed_{}.out'.format(
                    base_dir,
                    pretrain_config,
                    seed,
                    pretrain_config.replace('/','_'),
                    seed
                )
            else:
                # Stdout filename
                out_file='{}/{}_{}/seed_{}/{}_{}_seed_{}.out'.format(
                    base_dir,
                    pretrain_config,
                    finetune_config,
                    seed,
                    pretrain_config.replace('/','_'),
                    finetune_config,
                    seed
                )
            # Experiment model name
            model_name='classic_{}'.format(
                    pretrain_config.replace('/','_')
                    )
            
            experiment['bandit'] = None

        # Load
        if 'load' in experiment['training_script']:
            for param in experiment['training_script_params']:
                if '-p' in param:
                    pretrained_model=param.split('-p ')[1].split('./nlp_bandit_experiments/')[1]
                elif '-f' in param:
                    finetune_config=param.split('-f ')[1].split('fairseq_config/krylov/')[1]
                elif '-s' in param:
                    seed=param.split('-s ')[1]
            
            #Stdout filename
            out_file='{}/{}_{}/seed_{}/{}_{}_seed_{}.out'.format(
                base_dir,
                pretrained_model.replace('/','_').replace('.pt',''),
                finetune_config,
                seed,
                pretrained_model.replace('/','_').replace('.pt',''),
                finetune_config,
                seed
            )
            # Experiment model name
            model_name='load_{}'.format(
                    pretrained_model.replace('/','_').replace('.pt','').split('checkpoint/')[0],
                    )
            
            experiment['bandit'] = None
        
        # Bandit
        if 'bandit' in experiment['training_script']:
            finetune_config=''
            for param in experiment['training_script_params']:
                if '-b' in param:
                    bandit_config=param.split('-b ')[1].split('bandit_config/')[1]
                elif '-p' in param:
                    pretrain_config=param.split('-p ')[1].split('fairseq_config/krylov/')[1]
                elif '-f' in param:
                    finetune_config=param.split('-f ')[1].split('fairseq_config/krylov/')[1]
                elif '-s' in param:
                    seed=param.split('-s ')[1]
                
                
            # Pre-training only
            if finetune_config == '':
                #Stdout filename
                out_file='{}/{}/{}/seed_{}/{}_{}_seed_{}.out'.format(
                    base_dir,
                    pretrain_config,
                    bandit_config,
                    seed,
                    pretrain_config.replace('/','_'),
                    bandit_config.replace('/','_'),
                    seed
                )
                # Experiment model name
                model_name='bandit_{}_{}'.format(pretrain_config.replace('/','_'),bandit_config.replace('/','_'))
                # Bandit
                experiment['bandit']='{}/{}/{}/seed_{}/{}.pt'.format(
                    base_dir,
                    pretrain_config,
                    bandit_config,
                    seed,
                    os.path.basename(bandit_config)
                )
                
            # Pre-training and fine-tuning
            else:
                #Stdout filename
                out_file='{}/{}_{}/{}/seed_{}/{}_{}_{}_seed_{}.out'.format(
                    base_dir,
                    pretrain_config,
                    finetune_config,
                    bandit_config,
                    seed,
                    pretrain_config.replace('/','_'),
                    finetune_config,
                    bandit_config.replace('/','_'),
                    seed
                )
                # Experiment model name
                model_name='bandit_{}_{}_{}'.format(
                    pretrain_config.replace('/','_'),
                    finetune_config,
                    bandit_config.replace('/','_')
                )
                # Bandit
                experiment['bandit']='{}/{}_{}/{}/seed_{}/{}.pt'.format(
                    base_dir,
                    pretrain_config,
                    finetune_config,
                    bandit_config,
                    seed,
                    os.path.basename(bandit_config)
                )

        # Keep information
#         print(out_file)
        experiment['out_file']=out_file
        experiment['model_name']=model_name
        experiment['seed']=seed
#         # TODO: this is bandit specific now, should be probably moved to corresponding case above
        experiment['results']= Experiment(
                                    experiment['out_file'],
                                    experiment['bandit'],
        )

def load_experiment(exp_dir, exp='all', proc_funct=process_experiments):
    if exp=='all':
        exp_dict = {exp_i:[] for exp_i in get_nondir_files(exp_dir)}
    else:
        exp_dict={exp: []}
    for exp_i in exp_dict.keys():
        try:
            with open('{}/{}'.format(exp_dir, exp_i), 'rb') as f:
                load_exp=pickle.load(f)
            proc_funct(load_exp, exp_dir)
            exp_dict[exp_i] = load_exp
        except FileNotFoundError:
            print('File not found for {}'.format(exp_i))
            
    return exp_dict 


###################
## VISUALIZATION ##
###################

#
class Plot:

    def __init__(self, t, x_axis, y_axis):
        self.fig = go.Figure()

        self.fig = self.fig.update_layout(
            title=t,
            xaxis_title=x_axis,
            yaxis_title=y_axis,
        )

    def add_line(self, description, xy_dict, colour=False):

        if 'error' in xy_dict.keys():
            error_bars = xy_dict['error']
        else:
            error_bars = None

        if not colour:
            self.fig.add_trace(go.Scatter(x=xy_dict['x'], y=xy_dict['y'],
                                          mode='lines',
                                          name=description,
                                          error_y=dict(type='data', array=error_bars,visible=True)))
        else:
            self.fig.add_trace(go.Scatter(x=xy_dict['x'], y=xy_dict['y'],
                                          mode='lines',
                                          name=description,
                                          line=dict(color=colour),
                                          error_y=dict(type='data', array=error_bars,visible=True)))

    def show(self):
        self.fig.show()

    #### Specific plots


def plot_mprob_tv_loss(exp):
    # create plot
    plt_mprob = Plot('Mask prob during training', "iteraction", "mask_prob")

    # add masking probability
    plt_mprob.add_line("mprob", exp.get_bandit_m('mask_prob'))
    # add training loss
    plt_mprob.add_line("train_loss", exp.get_bandit_m('train_loss'))
    # add validation loss
    plt_mprob.add_line("val_loss", exp.get_bandit_m('val_loss'))

    plt_mprob.show()
