#!/usr/bin/python

# Imports: python modules
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_bandit_model_posterior(bandit, arm_space=None, context=None, plot_mode='mean', sample_shape=torch.Size(), plot_filename=None):
    # Decide on arm space to use
    if arm_space is None:
        arm_space=bandit.arm_space
    
    # Figure out arm space dimensionality
    d_arms=arm_space.shape[1]
    
    # For now, only 1D implemented
    assert d_arms==1, 'Only 1D arm plotting implemented'
    
    plt.figure()
    # Posterior mean
    if 'mean' in plot_mode:
        if context is None:
            posterior_mean=bandit.reward_model.mean(arm_space).detach()
        else:
            posterior_mean=bandit.reward_model.mean(context, arm_space).detach()
            
        plt.plot(
            arm_space[:,0],
            posterior_mean,
            'r',
            label='Model posterior'
            )
        
        # Plot std?
        if 'std' in plot_mode:
            if context is None:
                posterior_std=bandit.reward_model.standard_deviation(arm_space).detach()     
            else:
                posterior_std=bandit.reward_model.standard_deviation(context, arm_space).detach()
                
            plt.fill_between(
                arm_space[:,0],
                posterior_mean-posterior_std,
                posterior_mean+posterior_std,
                alpha=0.5,
                facecolor='r'
        )
    
    if 'sample' in plot_mode:
        if context is None:
            posterior_samples=bandit.reward_model.sample(arm_space, sample_shape).detach()   
        else:
            posterior_samples=bandit.reward_model.sample(context, arm_space, sample_shape).detach()
            
        plt.plot(
            arm_space[:,0],
            posterior_samples,
            'r',
            label='Model posterior samples'
            )
    
    plt.xlabel('a')
    plt.ylabel(r'$f(a)$')
    plt.title('Bandit model posterior')
    plt.autoscale(enable=True, tight=True)
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)        
    if plot_filename is None: 
        plt.show()
    else:
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        plt.close()
    

def plot_rewards(observed_rewards, optimal_expected_rewards, t_plot, plot_std=True, plot_filename=None):
    # rewards over time
    plt.figure()
    plt.plot(
            np.arange(t_plot),
            optimal_expected_rewards[0,:t_plot], 'k', label='Expected Optimal')
    plt.plot(
            np.arange(t_plot),
            observed_rewards.mean(axis=0)[:t_plot],
            'r',
            label='My_bandit'
        )
    if plot_std:
        plt.fill_between(
                np.arange(t_plot),
                observed_rewards.mean(axis=0)[:t_plot]-observed_rewards.std(axis=0)[:t_plot],
                observed_rewards.mean(axis=0)[:t_plot]+observed_rewards.std(axis=0)[:t_plot],
                alpha=0.5,
                facecolor='r'
        )
    plt.xlabel('t')
    plt.ylabel(r'$y_t$')
    plt.title('Rewards over time')
    plt.autoscale(enable=True, tight=True)
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_filename is None: 
        plt.show()
    else:
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        plt.close()

def plot_cumrewards(observed_rewards, optimal_expected_rewards, t_plot, plot_std=True, plot_filename=None):
    # rewards over time
    plt.figure()
    plt.plot(
            np.arange(t_plot),
            optimal_expected_rewards.cumsum(axis=1)[0,:t_plot], 'k', label='Expected Optimal')
    plt.plot(
            np.arange(t_plot),
            observed_rewards.cumsum(axis=1).mean(axis=0)[:t_plot],
            'r',
            label='My_bandit'
        )
    if plot_std:
        plt.fill_between(
                np.arange(t_plot),
                observed_rewards.cumsum(axis=1).mean(axis=0)[:t_plot]-observed_rewards.cumsum(axis=1).std(axis=0)[:t_plot],
                observed_rewards.cumsum(axis=1).mean(axis=0)[:t_plot]+observed_rewards.cumsum(axis=1).std(axis=0)[:t_plot],
                alpha=0.5,
                facecolor='r'
        )
    plt.xlabel('t')
    plt.ylabel(r'$\sum_{t}y_t$')
    plt.title('Cumulative rewards over time')
    plt.autoscale(enable=True, tight=True)
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_filename is None: 
        plt.show()
    else:
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        plt.close()
    
def plot_cumregret(observed_rewards, optimal_expected_rewards, t_plot, plot_std=True, plot_filename=None):
    # rewards over time
    plt.figure()
    regret=optimal_expected_rewards-observed_rewards
    plt.plot(
            np.arange(t_plot),
            regret.cumsum(axis=1).mean(axis=0)[:t_plot],
            'r',
            label='My_bandit'
        )
    if plot_std:
        plt.fill_between(
                np.arange(t_plot),
                regret.cumsum(axis=1).mean(axis=0)[:t_plot]-regret.cumsum(axis=1).std(axis=0)[:t_plot],
                regret.cumsum(axis=1).mean(axis=0)[:t_plot]+regret.cumsum(axis=1).std(axis=0)[:t_plot],
                alpha=0.5,
                facecolor='r'
        )
    plt.xlabel('t')
    plt.ylabel(r'$\sum_{t}r_t$')
    plt.title('Cumulative regret over time')
    plt.autoscale(enable=True, tight=True)
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    if plot_filename is None: 
        plt.show()
    else:
        plt.savefig(plot_filename, format='pdf', bbox_inches='tight')
        plt.close()

