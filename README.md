# Readme

This repository contains the code implementation of VAED-ensDAG.

## Data Preparation

The data required for this project can be obtained from the [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=633694461385244672). You will need to download and configure it in the project directory before running the experiments.

## Repository Structure

- `upstream.py`: The upstream code file. Contains causal discovery (FCI, PC, etc.), feature selection, graph construction, and model/network class encapsulations related to DAG-GNN and VAED/iVAE.
- `downstream.py`: The downstream code file. Contains the main execution entry (`main`), Graph Neural Network models (GraphSAGE, ECMPNN, GCN), traditional machine learning evaluation models (RF, MLP, LGB, XGB), as well as the prediction and evaluation pipelines. This file serves as the main entry point to run the system.
- `experiment.py`: The original unified execution code (a backup of the monolithic logic before it was split).
- `get_k3_latent.py`: Auxiliary script (for extracting latent variables/clustering auxiliary models when K=3).
- `run_avaed_ablations.py`: Script to control and batch-run AVAED ablation experiments.
- `prompt.txt` : Model prompts 
- `LICENSE` : Open source license files.

## Environment Requirements

The following are the core dependencies and tested versions in the `myCausalML` environment for this workspace:

- **Python** $\ge$ `3.11`
- **PyTorch** $\ge$ `2.0.0`
- **NumPy** $\ge$ `1.24.0` 
- **Pandas** $\ge$ `2.0.0`
- **NetworkX** $\ge$ `2.8.0`
- **Matplotlib** $\ge$ `3.5.0`
- **Scikit-Learn** $\ge$ `1.3.0`
- **XGBoost** $\ge$ `2.0.0`
- **Causal-Learn** $\ge$ `0.1.3`

It is recommended to use `conda` to install a PyTorch version that matches your CUDA hardware, and then install other dependencies via `pip` or `conda`.

