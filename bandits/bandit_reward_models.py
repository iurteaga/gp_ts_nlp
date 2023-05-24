#!/usr/bin/python
"""
@author: Iñigo Urteaga
"""

# Imports: python modules
import abc
# Science
import numpy as np
import torch
import gpytorch

# Debugging
import pdb

######## Generic Bandit Reward model definition ########
class ContinuousBanditRewardModel(abc.ABC,object):
    '''
        Abstract Class for Continuous Bandit reward models
    
        Attributes:
            
        
    '''
    @abc.abstractmethod
    def __init__(self):
        '''
            Initialize the Bandit object
        
            Args:
                None
        '''
        pass
    
    @abc.abstractmethod
    def update_model(self):
        '''
            Update the reward model
        
            Input:
                None
            Output:
                None
        '''
        pass
        
    @abc.abstractmethod
    def mean(self):
        '''
            Return the reward model's expected value
        
            Input:
                None
            Output:
                the reward model's expected value
        '''
        pass
    
    @abc.abstractmethod
    def standard_deviation(self):
        '''
            Return the reward model's standard deviation
        
            Input:
                None
            Output:
                the reward model's standard deviation
        '''
        pass
        
    @abc.abstractmethod
    def sample(self):
        '''
            Return a sample from the model's expected reward
        
            Input:
                None
            Output:
                a sample from the model's expected reward
        '''
        pass
        
######## Gaussian Process-based Bandit Reward model definition ########
class GPRewardModel(ContinuousBanditRewardModel):
    '''
        Class for Gaussian Proces Bandit reward models
    
        Attributes:
            
        
    '''
    
    def __init__(self, gp_model, likelihood_model, gp_training):
        '''
            Initialize the Bandit object
        
            Args:
                gp_model: assumed GP reward model
                likelihood_model: assumed GP reward model
                gp_training: GP related training options
        '''
        # Initialize Bandit
        super().__init__()
        
        # GP Attributes
        # Initial model is based on prior
        self.reward_model=gp_model
        # Given observation likelihood
        self.reward_model_likelihood=likelihood_model
        
        # GP training: params should be provided as dictionary
        self.gp_training=gp_training
    
    def _reward_model_posterior(self, arm_space):
        '''
            Compute the reward model's posterior predictive at arm_space
        
            Input:
                arm_space: arm space to compute model posterior over
                
            Output:
                predictive_posterior
        '''
        # Set into eval mode
        self.reward_model.eval()
            
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictive_posterior = self.reward_model(arm_space)
        
        return predictive_posterior
        
    def update_model(self, played_arms, observed_rewards):
        '''
            Update the reward model
        
            Input:
                played_arms
                observed_rewards
            Output:
                None
        '''
        # Instantiate the GP model with observed bandit history and likelihood
        self.reward_model.set_train_data(
                                inputs=played_arms,
                                targets=observed_rewards,
                                strict=False
                                )                        
        self.reward_model.train()
        
        # Instantiate the gp optimizer
        optimizer = self.gp_training['optimizer'](
                        self.reward_model.parameters(),
                        **self.gp_training['optimizer_params']
                        )
        
        # Instantiate "Loss" for GPs 
        gp_loss = self.gp_training['loss'](
                        self.reward_model_likelihood,
                        self.reward_model
                        )
        
        # Start training    
        print('| BANDIT | Reward model | GP model training started...')
        # Tmp variables
        n_iter=0
        prev_loss=0
        this_loss=np.inf
                
        # Iterate
        while (
                (n_iter < self.gp_training['n_train_max_iters']) 
                and 
                (abs(this_loss - prev_loss) >= self.gp_training['loss_epsilon']*abs(prev_loss))
                ):
                
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Output from model
            output = self.reward_model(played_arms)
            
            # Calc loss and backprop gradients
            loss = -gp_loss(output, observed_rewards)
            loss.backward()
            
            # Keep track of iterations
            n_iter+=1
            prev_loss=this_loss
            this_loss=loss.item()
            print('| BANDIT | Reward model | \tIteration {}/{} with loss={:.3f}'.format(
                n_iter + 1,
                self.gp_training['n_train_max_iters'],
                this_loss,
            ))
            
            optimizer.step()
        
        # Done training    
        print('| BANDIT | Reward model | GP model updated after {} iterations with loss={:.3f}'.format(
            n_iter, this_loss)
            )
    
    
    def mean(self, arm_space):
        '''
            Return the GP reward model's expected value
        
            Input:
                arm_space: arm space to compute model posterior over
            Output:
                posterior mean evaluated at arm_space
        '''
        
        return self._reward_model_posterior(arm_space).mean
    
    def standard_deviation(self, arm_space):
        '''
            Return the GP reward model's standard deviation
        
            Input:
                arm_space: arm space to compute model posterior over
            Output:
                posterior standard deviation evaluated at arm_space
        '''
 
        return self._reward_model_posterior(arm_space).stddev
        
    def sample(self, arm_space, sample_shape=torch.Size()):
        '''
            Return a sample from the model's expected reward
        
            Input:
                arm_space: arm space to compute model posterior over
            Output:
                a sample from the model's expected reward
        '''
        
        return self._reward_model_posterior(arm_space).sample(sample_shape)
        
######## Gaussian Process-based Contextual Bandit Reward model definition ########
class GPContextualRewardModel(ContinuousBanditRewardModel):
    '''
        Class for Gaussian Proces Contextual Bandit reward models
    
        Attributes:
            
        
    '''
    
    def __init__(self, gp_model, likelihood_model, gp_training):
        '''
            Initialize the Bandit object
        
            Args:
                gp_model: assumed GP reward model
                likelihood_model: assumed GP reward model
                gp_training: GP related training options
        '''
        # Initialize Bandit
        super().__init__()
        
        # GP Attributes
        # Initial model is based on prior
        self.reward_model=gp_model
        # Given observation likelihood
        self.reward_model_likelihood=likelihood_model
        
        # GP training: params should be provided as dictionary
        self.gp_training=gp_training
    
    def _reward_model_posterior(self, context, arm_space):
        '''
            Compute the reward model's posterior predictive at arm_space
        
            Input:
                context: observed context for the model posterior to consider
                arm_space: arm space to compute model posterior over
                
            Output:
                predictive_posterior
        '''        
        # We only observe one context value,
        #   yet to compute the covariance function over gp input space
        #   we need to have gp_input of same dim=0 as action space
        input_context = context[None,:]*torch.ones((arm_space.shape[0],1))
        gp_input=torch.cat(
                    (input_context,arm_space),
                    dim=1
                )
        
        # Set into eval mode
        self.reward_model.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictive_posterior = self.reward_model(gp_input)
        
        return predictive_posterior
        
    def update_model(self, observed_contexts, played_arms, observed_rewards):
        '''
            Update the reward model
        
            Input:
                observed_contexts
                played_arms
                observed_rewards
            Output:
                None
        '''
        # Instantiate the GP model with observed bandit history and likelihood
        observed_gp_input=torch.cat(
                            (observed_contexts,played_arms),
                            dim=1
                        )
        self.reward_model.set_train_data(
                                inputs=observed_gp_input,
                                targets=observed_rewards,
                                strict=False
                                )
        self.reward_model.train()
        
        # Instantiate the gp optimizer
        optimizer = self.gp_training['optimizer'](
                        self.reward_model.parameters(),
                        **self.gp_training['optimizer_params']
                        )
        
        # Instantiate "Loss" for GPs 
        gp_loss = self.gp_training['loss'](
                        self.reward_model_likelihood,
                        self.reward_model
                        )
        
        # Start training    
        print('| BANDIT | Reward model | GP model training started...')
        # Tmp variables
        n_iter=0
        prev_loss=0
        this_loss=np.inf
                
        # Iterate
        while (
                (n_iter < self.gp_training['n_train_max_iters']) 
                and 
                (abs(this_loss - prev_loss) >= self.gp_training['loss_epsilon']*abs(prev_loss))
                ):
                
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Output from model, given context-action input
            output = self.reward_model(observed_gp_input)
            
            # Calc loss and backprop gradients
            loss = -gp_loss(output, observed_rewards)
            loss.backward()
            
            # Keep track of iterations
            n_iter+=1
            prev_loss=this_loss
            this_loss=loss.item()
            print('| BANDIT | Reward model | \tIteration {}/{} with loss={:.3f}'.format(
                n_iter + 1,
                self.gp_training['n_train_max_iters'],
                this_loss,
            ))
            
            optimizer.step()
        
        # Done training    
        print('| BANDIT | Reward model | GP reward model updated after {} iterations with loss={:.3f}'.format(
            n_iter, this_loss)
            )
    
    
    def mean(self, context, arm_space):
        '''
            Return the GP reward model's expected value
        
            Input:
                context: observed context to compute model posterior over
                arm_space: arm space to compute model posterior over
            Output:
                posterior mean evaluated at arm_space
        '''
        
        return self._reward_model_posterior(context, arm_space).mean
    
    def standard_deviation(self, context, arm_space):
        '''
            Return the GP reward model's standard deviation
        
            Input:
                context: observed context to compute model posterior over
                arm_space: arm space to compute model posterior over
            Output:
                posterior standard deviation evaluated at arm_space
        '''
 
        return self._reward_model_posterior(context, arm_space).stddev
        
    def sample(self, context, arm_space, sample_shape=torch.Size()):
        '''
            Return a sample from the model's expected reward
        
            Input:
                context: observed context to compute model posterior over
                arm_space: arm space to compute model posterior over
            Output:
                a sample from the model's expected reward
        '''
        
        return self._reward_model_posterior(context, arm_space).sample(sample_shape)
