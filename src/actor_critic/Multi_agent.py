import numpy as np
from stable_baselines3 import A2C
import random
from env import Gym2OpEnv
import os

# Create multi-agent capable of all three features
Multi_agent_environment = Gym2OpEnv(act_type="mdisc", reduce_obs=False, acts_to_keep=["set_line_status", "set_bus", "sub_set_bus"])


# Load in pretrained agent for each action
set_line_status_env = Gym2OpEnv(act_type="mdisc", reduce_obs=False, acts_to_keep=["set_line_status"])
set_line_status_model = A2C.load("./actor_critic_models/set_line_status_Best_model.zip")
set_line_status_model.set_env(set_line_status_env)

set_bus_env = Gym2OpEnv(act_type="mdisc", reduce_obs=False, acts_to_keep=["set_bus"])
set_bus_model = A2C.load("./actor_critic_models/Best_model_set_bus.zip")
set_bus_model.set_env(set_bus_env)

sub_set_bus_env = Gym2OpEnv(act_type="mdisc", reduce_obs=False, acts_to_keep=["sub_set_bus"])
sub_set_bus_model = A2C.load("./actor_critic_models/Best_model_sub_set_bus.zip")
sub_set_bus_model.set_env(sub_set_bus_env)


# Run evaluation
os.makedirs('./results', exist_ok=True)
reward_log = "./results/multiagent_rewards.txt"
episode_len_log = "./results/multiagent_length.txt"
for i in range(0,1000):
    done = False
    obs, info = Multi_agent_environment.reset(options={"init ts": random.randint(60,1000)})
    set_line_status_env.reset()
    set_bus_env.reset()
    sub_set_bus_env.reset()
    ep_len = 0
    rewards = 0
    while done == False:
        set_line_action = set_line_status_model.predict(obs)
        set_bus_action = set_bus_model.predict(obs)
        sub_set_bus_action = sub_set_bus_model.predict(obs)
        full_action = np.concatenate((set_bus_action[0], set_line_action[0],sub_set_bus_action[0]))
        obs, reward, terminated, truncated, info = Multi_agent_environment.step(full_action)
        done = terminated       
        ep_len +=1
        rewards +=reward       
    with open(reward_log, 'a') as firstfile:  
        firstfile.write(str(rewards) + '\n')
    firstfile.close()
    with open(episode_len_log, 'a') as secondfile:  
        secondfile.write(str(ep_len)+ '\n')
    secondfile.close()
    







