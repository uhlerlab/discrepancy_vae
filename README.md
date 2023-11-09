# Discrepancy-based VAE

Code for paper: _Identifiability Guarantees for Causal Disentanglement from Soft Interventions (NeurIPS 2023)_

arXiv link: https://arxiv.org/abs/2307.06250

## Experiments on Biological Data

First download the dataset following instructions in `./data/README.md`.

Our model (**discrepancy-based VAE**) can be trained by running 
```
python run.py --device DEVICE
```
after replacing DEVICE with the supported compute (e.g., cuda:0). The **hyperparameters** are specified in `opts` (line 19 in `./src/run.py`). 

To run the **ablation studies**, set `--model cvae` for ours w/o discrepancy and `--model mvae` for ours w/o causal layer in the previous command.

Once training is done, **inference and sampling** can be done using functions in `./src/inference.py`. Examples using this can be found in `./notebooks/plot_samples.ipynb`, where we can reproduce all the figures and numbers in the paper. To reproduce the **learned programs and latent DAG**, use `./notebooks/check_DAG.ipynb`.

## Experiment on Toy Simulation

Generate sythentic data by running `./data/simulation/generate.py`. Then follow `./notebooks/run_simu.ipynb` to obtain the results.


## Results on Double-Node Interventions

All figures can be found in `./notebooks/figures`. For the trained models, these are uploaded to  due to space limit.