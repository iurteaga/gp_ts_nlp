#!/usr/bin/python

# Imports: python modules
import abc
# Debugging
import pdb
# Science
import torch

######## Generic Bandit Class definition ########
class Bandit(abc.ABC,object):
    '''
        Abstract Class for Bandits
    
        Attributes:
            reward_model: assumed reward model
            algorithm: string indicating what algorithm to execute
        
    '''
    
    def __init__(self, reward_model, algorithm, t_max=100):
        '''
            Initialize the Bandit object
        
            Args:
                reward_model: assumed reward model
                algorithm: string indicating what algorithm to execute
        '''
        # General Attributes
        self.reward_model=reward_model
        self.algorithm=algorithm
        self.t_max=t_max
        
        # Time of play
        self.t=0
        # History of rewards can be pre-allocated
        self.observed_rewards=torch.zeros(self.t_max)
        # History of played arms depends on discrete/continous action dimensionality
        self.played_arms=None
        
    def restart(self, t_max=None):
        '''
            Restart the Bandit to initial state
        
            Args:
                t_max: number of integers for new execution
        '''
        if t_max is not None:
            self.t_max=t_max
        
        # Time of play
        self.t=0
        # History of rewards can be pre-allocated
        self.observed_rewards=torch.zeros(self.t_max)
        # History of played arms depends on discrete/continous action dimensionality
        self.played_arms=None
    
    def update_history(self, played_arm, observed_reward):
        '''
            Keep track of arm/reward history and update the reward model accordingly
        
            Input:
                played_arm: played arm
                observed_reward: observed reward
            Output:
                None
        '''
        if self.t>=self.t_max:
            # Preallocate more space than initially thought
            self.played_arms=torch.cat(
                    (self.played_arms, torch.zeros_like(self.played_arms)),
                    dim=0
                    )
            self.observed_rewards=torch.cat(
                    (self.observed_rewards, torch.zeros_like(self.observed_rewards)),
                    dim=0
                    )
            self.t_max*=2
        
        # Update history
        self.played_arms[self.t]=played_arm
        self.observed_rewards[self.t]=observed_reward
        # Update bandit iteration count
        self.t+=1
        
        # Update model
        self.reward_model.update_model(
                self.played_arms[:self.t],
                self.observed_rewards[:self.t],
                )
    
    @abc.abstractmethod
    def next_action(self):
        '''
            Decide what arm to play next, based on implemented algorithm
        
            Input:
                None
            Output:
                played arm
        '''
        pass

######## Generic Contextual Bandit Class definition ########
class ContextualBandit(Bandit):
    '''
        Class for Contextual Bandits
    
        Attributes:
            reward_model: assumed reward model
            algorithm: string indicating what algorithm to execute
        
    '''
    
    def __init__(self, d_context, reward_model, algorithm, t_max=100):
        '''
            Initialize the Bandit object
        
            Args:
                d_context: dimensionality of the considered context
                reward_model: assumed reward model
                algorithm: string indicating what algorithm to execute
        '''
        # Initialize Bandit
        super().__init__(reward_model, algorithm, t_max)
        # Context size
        self.d_context=d_context
        
        # Time of play
        self.t=0
        
        # History of rewards can be pre-allocated
        self.observed_rewards=torch.zeros(self.t_max)
        # History of contexts can be pre-allocated
        self.observed_contexts=torch.zeros((self.t_max,d_context))
        # History of played arms depends on discrete/continous action dimensionality
        self.played_arms=None
    
    def restart(self, t_max=None):
        '''
            Restart the Bandit to initial state
        
            Args:
                t_max: number of integers for new execution
        '''
        if t_max is not None:
            self.t_max=t_max
        
        # Time of play
        self.t=0
        
        # History of rewards can be pre-allocated
        self.observed_rewards=torch.zeros(self.t_max)
        # History of contexts can be pre-allocated
        self.observed_contexts=torch.zeros((self.t_max,self.d_context))
        # History of played arms depends on discrete/continous action dimensionality
        self.played_arms=None
            
    def update_history(self, observed_context, played_arm, observed_reward):
        '''
            Keep track of arm/reward history and update the reward model accordingly
        
            Input:
                played_arm: played arm
                observed_reward: observed reward
            Output:
                None
        '''
        if self.t>=self.t_max:
            # Preallocate more space than initially thought
            self.observed_contexts=torch.cat(
                    (self.observed_contexts, torch.zeros_like(self.observed_contexts)),
                    dim=0
                    )
            self.played_arms=torch.cat(
                    (self.played_arms, torch.zeros_like(self.played_arms)),
                    dim=0
                    )
            self.observed_rewards=torch.cat(
                    (self.observed_rewards, torch.zeros_like(self.observed_rewards)),
                    dim=0
                    )
            self.t_max*=2
        
        # Update history
        self.observed_contexts[self.t]=observed_context
        self.played_arms[self.t]=played_arm
        self.observed_rewards[self.t]=observed_reward
        # Update bandit iteration count
        self.t+=1
        
        # Update model
        self.reward_model.update_model(
                self.observed_contexts[:self.t],
                self.played_arms[:self.t],
                self.observed_rewards[:self.t],
                )
        
######## Discrete Bandit Class definition ########
class DiscreteArmBandit(Bandit):
    '''
        Class for Discrete Bandits
    
        Attributes:
            
            reward_model: assumed reward model
            algorithm: string indicating what algorithm to execute
        
    '''
    
    def __init__(self, A, reward_model, algorithm):
        '''
            Initialize the Bandit object
        
            Args:
                A: number of discrete arms
                reward_model: assumed reward model
                algorithm: string indicating what algorithm to execute
        '''
        # Initialize Bandit
        super().__init__(reward_model, algorithm)
        # Number of arms
        self.A=A
        
        # History of arms, pre-allocate based on dimensionality
        self.played_arms=torch.zeros(self.t_max,self.A)
    
    def restart(self, t_max=None):
        '''
            Restart the Bandit to initial state
        
            Args:
                t_max: number of integers for new execution
        '''
        super().restart(t_max)
        
        # History of arms, pre-allocate based on dimensionality
        self.played_arms=torch.zeros(self.t_max,self.A)
        
    def next_action(self):
        '''
            Decide what arm to play next, based on implemented algorithm
        
            Input:
                None
            Output:
                played arm
        '''
        # Decide next action
        with torch.no_grad():
            a=None
            if self.algorithm['name'] == 'ThompsonSampling':
                a=torch.argmax(
                    self.reward_model.sample()
                    , axis=1)
            if self.algorithm['name'] == 'UCB':
                a=torch.argmax(
                    self.reward_model.mean()
                    + torch.sqrt(self.algorithm['beta'](self.t)) 
                        * self.reward_model.standard_deviation()
                    , axis=1)
        
        return a

######## Discrete Contextual Bandit Class definition ########
class DiscreteArmContextualBandit(ContextualBandit):
    '''
        Class for Discrete Contextual Bandits
    
        Attributes:
            
            reward_model: assumed reward model
            algorithm: string indicating what algorithm to execute
        
    '''
    
    def __init__(self, d_context, A, reward_model, algorithm):
        '''
            Initialize the Bandit object
        
            Args:
                d_context: dimensionality of the considered context
                A: number of discrete arms
                reward_model: assumed reward model
                algorithm: string indicating what algorithm to execute
        '''
        # Initialize Bandit
        super().__init__(d_context, reward_model, algorithm)
        # Number of arms
        self.A=A
        
        # History of arms, pre-allocate based on dimensionality
        self.played_arms=torch.zeros(self.t_max,self.A)
    
    def restart(self, t_max=None):
        '''
            Restart the Bandit to initial state
        
            Args:
                t_max: number of integers for new execution
        '''
        super().restart(t_max)
        
        # History of arms, pre-allocate based on dimensionality
        self.played_arms=torch.zeros(self.t_max,self.A)
        
    def next_action(self, context):
        '''
            Decide what arm to play next, based on implemented algorithm
        
            Input:
                context: observed context for the next action to take
            Output:
                arm to play next
        '''
        
        # Decide next action
        with torch.no_grad():
            a=None
            if self.algorithm['name'] == 'ThompsonSampling':
                a=torch.argmax(
                    self.reward_model.sample(context)
                    , axis=1)
            if self.algorithm['name'] == 'UCB':
                a=torch.argmax(
                    self.reward_model.mean(context)
                    + torch.sqrt(self.algorithm['beta'](self.t)) 
                        * self.reward_model.standard_deviation(context)
                    , axis=1)
        
        return a
    
    
 
######## Continuous Bandit Class definition ########
class ContinuousArmBandit(Bandit):
    '''
        Class for Continuous arm Bandits
    
        Attributes:
            
            reward_model: assumed reward model
            algorithm: string indicating what algorithm to execute
        
    '''
    
    def __init__(self, arm_space, reward_model, algorithm):
        '''
            Initialize the Bandit object
        
            Args:
                arm_space: set/space of countinuous arms (i.e., the space of arms) 
                reward_model: assumed reward model
                algorithm: string indicating what algorithm to execute
        '''
        # Initialize Bandit
        super().__init__(reward_model, algorithm)
        # Arm space: per-arm continuous space times d_arm
        self.arm_space=arm_space
        
        # History of arms, pre-allocate based on dimensionality
        self.played_arms=torch.zeros(self.t_max,self.arm_space.shape[1])
    
    def restart(self, t_max=None):
        '''
            Restart the Bandit to initial state
        
            Args:
                t_max: number of integers for new execution
        '''
        super().restart(t_max)
        
        # History of arms, pre-allocate based on dimensionality
        self.played_arms=torch.zeros(self.t_max,self.arm_space.shape[1])
        
    def next_action(self, arm_space=None):
        '''
            Decide what arm to play next, based on implemented algorithm
        
            Input:
                arm_space: arm space to compute model posterior over
                    Default is None, so that self.arm_space is used
                        the goal is to add future flexibility/coarseness
            Output:
                arm to play next
        '''
        # Decide on arm space to use
        if arm_space is None:
            arm_space=self.arm_space
        
        # Decide next action
        with torch.no_grad():
            if self.algorithm['name'] == 'ThompsonSampling':
                a_max_idx=torch.argmax(
                    self.reward_model.sample(arm_space)
                    , axis=0)
            if self.algorithm['name'] == 'UCB':
                a_max_idx=torch.argmax(
                    self.reward_model.mean(arm_space) 
                    + torch.sqrt(self.algorithm['beta'](self.t)) 
                        * self.reward_model.standard_deviation(arm_space)
                    , axis=0)
        
        # Return selected action
        return self.arm_space[a_max_idx]

######## Continuous Contextual Bandit Class definition ########
class ContinuousArmContextualBandit(ContextualBandit):
    '''
        Class for Continuous arm Contextual Bandits
    
        Attributes:
            
            reward_model: assumed reward model
            algorithm: string indicating what algorithm to execute
        
    '''
    
    def __init__(self, d_context, arm_space, reward_model, algorithm):
        '''
            Initialize the Bandit object
        
            Args:
                d_context: dimensionality of the considered context
                arm_space: set/space of countinuous arms (i.e., the space of arms) 
                reward_model: assumed reward model
                algorithm: string indicating what algorithm to execute
        '''
        # Initialize Bandit
        super().__init__(d_context, reward_model, algorithm)
        
        # Arm space: per-arm continuous space times d_arm
        self.arm_space=arm_space
        
        # History of arms, pre-allocate based on dimensionality
        self.played_arms=torch.zeros(self.t_max,self.arm_space.shape[1])
    
    def restart(self, t_max=None):
        '''
            Restart the Bandit to initial state
        
            Args:
                t_max: number of integers for new execution
        '''
        super().restart(t_max)
        
        # History of arms, pre-allocate based on dimensionality
        self.played_arms=torch.zeros(self.t_max,self.arm_space.shape[1])    
    
    def next_action(self, context, arm_space=None):
        '''
            Decide what arm to play next, based on implemented algorithm
        
            Input:
                context: observed context for the next action to take
                arm_space: arm space to consider for next action
                    Default is None, so that self.arm_space is used
                        the goal is to add future flexibility/coarseness
            Output:
                arm to play next
        '''
        # Decide on arm space to use
        if arm_space is None:
            arm_space=self.arm_space
            
        # Decide next action, given context
        with torch.no_grad():
            if self.algorithm['name'] == 'ThompsonSampling':
                a_max_idx=torch.argmax(
                    self.reward_model.sample(context, arm_space)
                    , axis=0)
            if self.algorithm['name'] == 'UCB':
                a_max_idx=torch.argmax(
                    self.reward_model.mean(context, arm_space) 
                    + torch.sqrt(self.algorithm['beta'](self.t)) 
                        * self.reward_model.standard_deviation(context, arm_space)
                    , axis=0)
        
        # Return selected action
        return self.arm_space[a_max_idx]
     
