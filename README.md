## 1. Setup and installation

In order to replicate the results obtained, we recommend creating a $\texttt{conda}$ environment as follows

```bash
conda create -n grid2op_env python=3.8
conda activate grid2op_env
```

and installing all dependencies as:

```bash
conda install pytorch torchvision torchaudio -c pytorch
pip install stable-baselines3 wandb gymnasium grid2op lightsim2grid matplotlib
```

To deactivate the environment, run:
```bash
conda deactivate
```

## 2. Code structure

## 3. Running agents