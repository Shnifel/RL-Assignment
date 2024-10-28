#!/bin/sh 

cd src/option_critic

# Baselines
python train.py --exp baseline_cont --act_type cont --cull-obs False
python train.py --exp baseline_disc --act_type disc --cull-obs False

# Observation culling
python train.py --exp pruned_obs_space_baseline --act_type cont

# Attention gating
python train.py --exp attention_2 --act_type cont --use-attention True
python train.py --exp attention_4 --act_type cont --use-attention True --num-options 4
python train.py --exp attention_6 --act_type cont --use-attention True --num-options 6

# Reward shaping
python train.py --exp reward_shaping --act_type cont --use-attention True --reward-shaping True

# Advance Q-Network
python train.py --exp changed_q_net --act_type cont --use-attention True --advanced-q-net True

# Actions reparametersiation
python train.py --exp reparam_actions --act_type cont --use-attention True --advanced-q-net True --norm-actions True

