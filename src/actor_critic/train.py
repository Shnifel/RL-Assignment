from env import Gym2OpEnv
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
import os
import csv
import numpy as np
import torch
import joblib
from gymnasium.spaces import Box
from gym import Wrapper
import argparse


global n_step
parser = argparse.ArgumentParser(description="A2C Grid2Op")

parser.add_argument('--act_type', default='mdisc', help='One of disc mdisc cont')
parser.add_argument('--acts_to_keep', nargs="+", default=None, help='Actions to keep')
parser.add_argument('--reduce_obs', type = bool, default=False, help='Reduce observations or not')
parser.add_argument('--run_name', default='baseline', help='Name of run')

# Tensorboard callback for analysing and tracking actions - debugging
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.log_dir = "./logs"
        self.actions_dir = "./logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.model_save_path = "models/Best_model"

        self.file_path = os.path.join(self.log_dir, "Episode rewards.csv")
        self.action_path = os.path.join(self.log_dir, "Episode Ending actions.csv")
        self.okay_action = os.path.join(self.log_dir, "Good actions.csv")

        with open(self.file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode Reward"])

        with open(self.action_path, "w", newline="") as z:
                writer = csv.writer(z)

        with open(self.okay_action, "w", newline="") as l:
                    writer = csv.writer(l)

        self.episode_rewards = 0  
        self.episode_num = 0 
        self.ep_average = 0
        self.accumulated_reward = 0 
        self.ep_actions = []
        self.ep_rewards = []
        self.n_step=0
        self.max_return = 0
        self.prev_action = []
        self.prev_state = []

    def _on_step(self):

        reward = self.locals.get("rewards", [0])[0]
        action = self.locals.get("clipped_actions", [0])[0]
        self.episode_rewards += reward
        self.ep_actions.append(action)
        self.ep_rewards.append(reward)
        
        done = self.locals.get("dones", [False])[0]
        
        if done:
            
            with open(self.file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.episode_rewards])
   
            with open(self.action_path, "a", newline="") as z:
                writer = csv.writer(z)
                writer.writerow([action])
                
            if self.episode_rewards > self.max_return:
                sb3_algo1.save(self.model_save_path)
                self.max_return = self.episode_rewards
            
            if self.episode_rewards >= 1000:
                with open(self.okay_action, "a", newline="") as l:
                    writer = csv.writer(l)
                    for i in range(len(self.ep_actions)-1):
                        writer.writerow(self.ep_actions[i])

            self.episode_rewards = 0
            self.ep_actions = []
            self.ep_rewards= []
            self.n_step+=1
        
        return True



if __name__ == "__main__":

    # Env setup
    args = parser.parse_args()
    n_step = 0
    env = Gym2OpEnv(act_type=args.act_type, reduce_obs=args.reduce_obs, acts_to_keep=args.acts_to_keep)
    log_dir = f"./logs/"
    model_save_path = f"./actor_critic_models/{args.run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    sb3_algo1 = A2C("MlpPolicy", env, verbose=0, tensorboard_log=log_dir)

    # Model training
    sb3_algo1.learn(
        total_timesteps=int(1e6), 
        progress_bar=True,
        callback=[TensorboardCallback()]
    )
    sb3_algo1.save(model_save_path)

   
