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

    To execute GP-TS $\psi=\left(\rho, \gamma, \lambda\right): where the bandit optimizes over all MLM dynamic masking hyperparameters.
    The bandit search space is a three-dimensional hypercube $\Psi$,
    with no previous expert guidance on hyperparameter selection.
    
- bandit_config/b_mprob_005_050/b_gp_zero_rbf_pretrain_val_loss_delta
    
    To execute GP-TS $\rho$: where the bandit arm is the masking probability $\rho$ of replacing an input token with the \textit{mask} token


###

## Acknowledgements

Work partially funded by eBay's Research and University Partnership for Technology (eRUPT) program.
