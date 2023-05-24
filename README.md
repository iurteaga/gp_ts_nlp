# gp_ts_nlp

A Bayesian optimization framework for efficient pre-training of Transformer-based language models.

## Related publications

- ACL 2023 Findings

- NeurIPS 2022 Workshop onGaussian Processes, Spatiotemporal Modeling, and Decision-making Systems

    - [Gaussian Process Thompson sampling for Bayesian optimization of dynamic masking-based language model pre-training](https://gp-seminar-series.github.io/neurips-2022/assets/camera_ready/26.pdf)

- EMNLP 2022 Workshop on Novel Ideas in Learning-to-Learn through Interaction (NILLI)

    - [Thompson sampling for interactive Bayesian optimization of dynamic masking-based language model pre-training](https://www.cs.mcgill.ca/~pparth2/nilli_workshop_2022/accepted-papers/3.pdf)


## Directories

### arxiv_doc

Directory with the manuscript as in arxiv 

### bandit_config

Directory where configuration files for the bandit agents is kept.

Two examples are provided:

- bandit_config/b_all_mprob_005_050/b_gp_zero_rbf_pretrain_val_loss_delta

    To execute GP-TS $\psi=\left(\rho, \gamma, \lambda\right)$: where the bandit optimizes over all MLM dynamic masking hyperparameters.
    The bandit search space is a three-dimensional hypercube $\Psi$,
    with no previous expert guidance on hyperparameter selection.
    
- bandit_config/b_mprob_005_050/b_gp_zero_rbf_pretrain_val_loss_delta
    
    To execute GP-TS $\rho$: where the bandit arm is the masking probability $\rho$ of replacing an input token with the mask token

### bandits

Directory with the source code where different bandit agents are implemented.

- bandits/bandits.py is the main class for bandit agents
- bandits/bandit_reward_models.py is where the bandit reward models are defined
- bandits/gp_models.py is where Gaussian process models for bandit reward functions are defined
- bandits/example_notebooks contains example notebooks on how to use the source code to run bandit simulations

### datasets

Directory where to keep the datasets for pre-training and fine-tuning (not included in repository)

### fairseq_hydra

Directory where hydra config files for Fairseq's execution are provided

- fairseq_hydra/p_e1.yaml
    
    Example config that runs Fairseq for 1 epoch, i.e., 1 interaction equals 1 epoch

- fairseq_hydra/p_u100.yaml
    
    Example config that runs Fairseq for 100 updates, i.e., 1 interaction equals 100 gradient updates.
    
- fairseq_hydra/f_e10_glue.yaml
    
    Example configs to fine-tune Fairseq models in Glue tasks
    
NOTE: The above scripts need to be personalized to each user's server configuration and capabilities regarding directory structure and computational resources (memory and GPU) available.

### nlp_bandit_experiments

Directory where the results of run experiments are saved to (not included in the repository)

### nlp_bandit_scripts

Directory where auxiliary source code is provided for initializing/processing/plotting of the framework

- nlp_bandit_scripts/init_gp_bandit.py is used to initialize a GP-based bandit agent 
- nlp_bandit_scripts/bandit_utils.py provides some additional utility functions for the implemented bandit agent
- nlp_bandit_scripts/bandit_update_fairseq_hydra.py provides utility functions to update fairseq's hydra config files directly from bandit posteriors
- nlp_bandit_scripts/load_experiment.py provides utilities to load and process the experiment's output files
- nlp_bandit_scripts/plotting_functions.py contains some useful plotting functions

### shell_scripts

Shell scripts to execute the GP-TS framework for TLM pre-training optimization 

- shell_scripts/bandit_roberta_hydra.sh

    Runs GPTS with RoBERTa, i.e., the pre-training of RoBERTA within an interactive framework where hyperparameter selection is guided by GPTS
     
- shell_scripts/classic_roberta_hydra_interactions.sh
    
    Runs RoBERTa pre-training, i.e., pre-raining of RoBERTA within an interactive framework with no hyperparameter editing

- shell_scripts/random_roberta_hydra.sh

    Runs pre-training of RoBERTA within an interactive framework with randomly selected hyperparameters

- shell_scripts/load_roberta_finetune_hydra.sh

    Loads a RoBERTa model from a checkpoint saved at a specified path, and fine-tunes it based on provided config files

NOTE: The above scripts need to be personalized to each user's server configuration regarding directory structure and python environment set up.
    
## Acknowledgements

Work partially funded by eBay's Research and University Partnership for Technology (eRUPT) program.
