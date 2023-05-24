#!/usr/bin/python
"""
@author: Iñigo Urteaga
"""

# Imports: python modules
import abc
# Science
import torch
import numpy as np
import scipy.stats as stats
# Debugging
import pdb

class ContinuousLinearGaussianBanditModel():
    '''
        Linear Bandit with Gaussian rewards
    
        Attributes:
            
        
    '''
    def __init__(self, slope, intercept, noise_var):
        '''
            Initialize the Bandit object
        
            Args:
                None
        '''
        self.slope=slope
        self.intercept=intercept
        self.noise_var=noise_var
    
    def mean(self,a):
        '''
            Return the environments expected value
        
            Input:
                
            Output:
                the environments reward expected value
        '''
        return self.intercept + self.slope * a
    
    def optimal_arm(self,a):
        '''
            Return the environments optimal arm
        
            Input:
                
            Output:
                the environments reward expected value
        '''
        # Find optimal arm index
        a_ind=torch.argmax(self.intercept + self.slope * a)
        # Return arm value
        return a[a_ind]
        
    def play(self,a):
        '''
            Return a reward sample from the environment, given action
        
            Input:
                action
            Output:
                a sample reward from the environment
        '''
        if self.noise_var > 0:
            noise=stats.norm.rvs(loc=0,scale=np.sqrt(self.noise_var))
        else:
            noise=0
        
        return self.intercept + self.slope * a + noise


class ContinuousLinearBernoulliBanditModel():
    '''
        Linear Bandit reward, with Bernoulli outcomes
    
        Attributes:
            
        
    '''
    def __init__(self, slope, intercept):
        '''
            Initialize the Bandit object
        
            Args:
                None
        '''
        self.slope=slope
        self.intercept=intercept
    
    def mean(self,a):
        '''
            Return the environments expected value
        
            Input:
                
            Output:
                the environments reward expected value
        '''
        return self.intercept + self.slope * a
    
    def optimal_arm(self,a):
        '''
            Return the environments optimal arm
        
            Input:
                
            Output:
                the environments reward expected value
        '''
        # Find optimal arm index
        a_ind=torch.argmax(self.intercept + self.slope * a)
        # Return arm value
        return a[a_ind]
        
    def play(self,a):
        '''
            Return a reward sample from the environment, given action
        
            Input:
                action
            Output:
                a sample reward from the environment
        '''
        
        return stats.bernoulli.rvs(self.intercept + self.slope * a)

class ContinuousContextualLinearGaussianBanditModel():
    '''
        Linear Bandit with Gaussian rewards
    
        Attributes:
            
        
    '''
    def __init__(self, slope, intercept, noise_var):
        '''
            Initialize the Bandit object
        
            Args:
                None
        '''
        self.slope=slope
        self.intercept=intercept
        self.noise_var=noise_var
    
    def mean(self,a,context):
        '''
            Return the environments expected value
        
            Input:
                a: action space to evaluate
                context:
            Output:
                the environments reward expected value
        '''
        return torch.einsum(
                    '...d,...d->...',
                    context,
                    self.intercept + self.slope * a
                    )
    
    def optimal_arm(self,a,context):
        '''
            Return the environments optimal arm
        
            Input:
                
            Output:
                the environments reward expected value
        '''
        # Find optimal arm index
        a_ind=torch.argmax(
                torch.einsum(
                    '...d,...d->...',
                    context,
                    self.intercept + self.slope * a
                    ),
                axis=-1 # Assuming context's first dimension is t
                )
        # Return arm value
        return a[a_ind]
        
    def play(self,a,context):
        '''
            Return a reward sample from the environment, given action
        
            Input:
                action
            Output:
                a sample reward from the environment
        '''
        if self.noise_var > 0:
            noise=stats.norm.rvs(loc=0,scale=np.sqrt(self.noise_var))
        else:
            noise=0
        
        return torch.einsum(
                    '...d,...d->...',
                    context,
                    self.intercept + self.slope * a
                    ) + noise


class ContinuousContextualLinearBernoulliBanditModel():
    '''
        Linear Bandit reward, with Bernoulli outcomes
    
        Attributes:
            
        
    '''
    def __init__(self, slope, intercept):
        '''
            Initialize the Bandit object
        
            Args:
                None
        '''
        self.slope=slope
        self.intercept=intercept
    
    def mean(self,a,context):
        '''
            Return the environments expected value
        
            Input:
                
            Output:
                the environments reward expected value
        '''
        return torch.einsum(
                    '...d,...d->...',
                    context,
                    self.intercept + self.slope * a
                    )
    
    def optimal_arm(self,a,context):
        '''
            Return the environments optimal arm
        
            Input:
                
            Output:
                the environments reward expected value
        '''
        # Find optimal arm index
        a_ind=torch.argmax(
                torch.einsum(
                    '...d,...d->...',
                    context,
                    self.intercept + self.slope * a
                    ),
                axis=-1 # Assuming context's first dimension is t
                )
        # Return arm value
        return a[a_ind]
        
    def play(self,a,context):
        '''
            Return a reward sample from the environment, given action
        
            Input:
                action
            Output:
                a sample reward from the environment
        '''
        
        return stats.bernoulli.rvs(
                    torch.einsum(
                    '...d,...d->...',
                    context,
                    self.intercept + self.slope * a
                    )
                )
        
