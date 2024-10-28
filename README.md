## 1. Setup and installation

In order to replicate the results obtained, we recommend creating a $\texttt{conda}$ environment as follows

```bash
conda create -n grid2op_env python=3.11
conda activate grid2op_env
```

and installing all dependencies as:

```bash
conda install pytorch torchvision torchaudio -c pytorch
pip install stable-baselines3 wandb gymnasium grid2op lightsim2grid matplotlib pandas
```

To deactivate the environment, run:
```bash
conda deactivate
```

## 2. Code structure

```
src                                             
├── actor_critic                                 -> A2C Implementation
|   └── actor_critic_models                          Containts actor-critic models from training
|   └── Multi agent evaluation logs                  Containts the logs used to evlauate the Mulit-agent
|   └── env.py                                       Grid2Op environment configuration
|   └── eval.ipynb                                   Jupyter Notebook  visualizes results from logs   
|   └── train.py                                     Main file from which models are trained
|   └── utils                                        Folder containing utility scripts
|       └── Extract_csv.py                           Extracts logging results from tensorboard files
|       └── study_agent.py                           Used to Evaluate agents after training
└── option_critic                                -> Option-Critic Implementation
|   └── env.py                                       Grid2Op environment configuration
|   └── train.py                                     Main file from which models are trained
|   └── option_critic.py                             Adapted Option-Critic Network Implementation
|   └── experience_replay.py                         Experience replay for training critic
|   └── logger.py                                    Logging training rewards and losses to tensorboard
|   └── eval.py                                      Evaluates runs (Strictly works with given run file)
run_ac.sh                                        -> Training all ac iterations
run_oc.sh                                        -> Training all oc iterations
```

## 3. Running agents

### Actor-Critic

### Option-Critic

Our Option-Critic implementation, as detailed in the report, is adapted from: https://github.com/lweitkamp/option-critic-pytorch

We provide a single run script ```run_oc.sh``` in order to train and run each iteration, as well as evaluate and reproduce all the graphs in the report, this should be run from the root directory. We note that ```eval.py``` depends on this naming convention. 

If you would like to train your own agent, then the following commmand should be run:

```bash
cd src/option_critic
python train.py 
```

Run ```python train.py --help``` to get a comprehensive list of arguments.
