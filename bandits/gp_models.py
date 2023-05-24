#!/usr/bin/python

# Imports: python modules
import pdb
# Gaussian process with pytorch
import gpytorch

# Exact (non-contextual) GP model class
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, 
                    a, y, 
                    mean_function, kernel_function,
                    likelihood
                ):
        # Init Exact GP model
        super(ExactGPModel, self).__init__(a, y, likelihood)
        
        # Init all modules
        self.mean_module = mean_function
        self.covar_module = kernel_function

    def forward(self, a):
        mean_a = self.mean_module(a)
        covar_a = self.covar_module(a)
        return gpytorch.distributions.MultivariateNormal(mean_a, covar_a)

# Exact Contextual GP model class
class ExactContextualGPModel(gpytorch.models.ExactGP):
    def __init__(self,
                    gp_input, y,
                    d_context,
                    mean_functions, kernel_functions, action_context_composition,
                    likelihood
                ):
        # Init Exact GP model
        super(ExactContextualGPModel, self).__init__(gp_input, y, likelihood)
        
        # Figure out dimensionality of context and actions
        self.d_context=d_context
        
        # Init all modules
        self.mean_modules = mean_functions
        self.covar_modules = kernel_functions
        self.action_context_composition= action_context_composition
    
    def forward(self, gp_input):        
        if self.action_context_composition is None:
            # Action and context are modeled jointly
            mean=self.mean_modules['joint'](gp_input)
            covar=self.covar_modules['joint'](gp_input)
        
        else:
            # Action and context are modeled separately
            # Separate context and actions
            x = gp_input[:,:self.d_context]
            a = gp_input[:,self.d_context:]
            
            # Context Mean
            mean_x = self.mean_modules['context'](x)
            # Action Mean
            mean_a = self.mean_modules['action'](a)
            
            # Context kernel
            covar_x = self.covar_modules['context'](x)
            # Action kernel
            covar_a = self.covar_modules['action'](a)
            
            # Combine
            if self.action_context_composition == 'add':
                mean = mean_x + mean_a
                covar = covar_x + covar_a
            elif self.action_context_composition == 'product':
                mean = mean_x * mean_a
                covar = covar_x * covar_a
            else:
                raise ValueError('Action/context composition={} not implemented yet'.format(self.action_context_composition))
        
        # Return Multivariate Normal with computed mean and covariance
        return gpytorch.distributions.MultivariateNormal(mean, covar)

