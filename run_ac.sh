#!/bin/sh 

cd src/actor_critic

# Baselines
python train.py --run_name baseline_cont --act_type cont
python train.py --run_name baseline_disc --act_type disc 
python train.py --run_name baseline_mdisc --act_type mdisc 

# Reduced observations
python train.py --run_name reduced_obs --act_type mdisc 

# Training individual for multi agent
python train.py --run_name set_bus --act_type mdisc --acts_to_keep set_bus
python train.py --run_name set_line_status --act_type mdisc --acts_to_keep set_line_status
python train.py --run_name sub_set_bus --act_type mdisc --acts_to_keep sub_set_bus

# Multi agent evaluation
python multi_agent.py