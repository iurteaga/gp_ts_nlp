#!/usr/bin/python
import gpytorch
from configparser import ConfigParser 

### Cast dictionary values
def cast_dict_values(dict, dtype):
    '''
        Input:
            dict: dictionary for which to cast values
            dtype: what data-type to cast values to
        Output:
            dict: dictionary with casted values
    '''
    return {key:dtype(value) for (key,value) in dict.items()}
    
### Load gp functions into dictionary
def load_gp_functions_to_dict(config_parser, config_section):
    functions = {}
    for option in config_parser.options(config_section):
        functions[option]=eval(config_parser.get(config_section, option))
        
    return functions
