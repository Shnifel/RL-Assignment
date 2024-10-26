import gymnasium as gym
from gymnasium.spaces.box import Box
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from grid2op.gym_compat import DiscreteActSpace
from grid2op.gym_compat import MultiDiscreteActSpace
from lightsim2grid import LightSimBackend
import matplotlib.pyplot as plt
import numpy as np
import joblib
from stable_baselines3 import A2C
from gymnasium.spaces import MultiDiscrete
from grid2op.gym_compat.utils import ActType
import random

class Gym2OpEnv(gym.Env):
    def __init__(
            self,action_space
    ):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########
        self._gym_env = gym_compat.GymEnv(self._g2op_env)
        self.setup_observations()
        if action_space == "set_line_status":
            self.setup_actions_set_line_status()
        elif action_space == "set_bus":
            self.setup_actions_set_bus()
        elif action_space == "mda":
            
            self.setup_actions_all_actions()
        elif action_space =="discrete":
            self.setup_actions_discrete()
        elif action_space =="cont":
            self.setup_actions_all_cont()
        else:
            self.setup_actions_sub_set_bus()

    def setup_observations(self):

        attributes_to_keep = [
            "a_ex", "a_or", "active_alert", "actual_dispatch", "alert_duration",
            "attack_under_alert", "attention_budget", "current_step", "curtailment",
            "curtailment_limit", "curtailment_limit_effective", "curtailment_limit_mw",
            "curtailment_mw", "day", "day_of_week", "delta_time", "duration_next_maintenance",
            "gen_margin_down", "gen_margin_up", "gen_p", "gen_p_before_curtail", "gen_q",
            "gen_theta", "gen_v", "hour_of_day", "is_alarm_illegal", "last_alarm",
            "line_status", "load_p", "load_q", "load_theta", "load_v", "max_step",
            "minute_of_hour", "month", "p_ex", "p_or", "prod_p", "prod_q", "prod_v",
            "q_ex", "q_or", "rho", "storage_charge", "storage_power", "storage_power_target",
            "storage_theta", "target_dispatch", "thermal_limit", "theta_ex", "theta_or",
            "time_before_cooldown_line", "time_before_cooldown_sub", "time_next_maintenance",
            "time_since_last_alarm", "time_since_last_alert", "time_since_last_attack",
            "timestep_overflow", "topo_vect", "total_number_of_alert", "v_ex", "v_or",
            "was_alarm_used_after_game_over", "was_alert_used_after_attack", "year"
        ]
       
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = gym_compat.BoxGymObsSpace(self._g2op_env.observation_space, attr_to_keep=attributes_to_keep)
        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                     low=self._gym_env.observation_space.low,
                                     high=self._gym_env.observation_space.high)
        


    def setup_actions_set_line_status(self):
        self._gym_env.action_space.close()
        

        act_attr_to_keep = [
           "set_line_status"
            ]

        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space, attr_to_keep = act_attr_to_keep)
        self.action_space = self._gym_env.action_space
        

    def setup_actions_set_bus(self):
        self._gym_env.action_space.close()
    

        act_attr_to_keep = [
            "set_bus"
            ]

        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space, attr_to_keep = act_attr_to_keep)
        self.action_space = self._gym_env.action_space

    def setup_actions_sub_set_bus(self):
        self._gym_env.action_space.close()
        act_attr_to_keep = [
            "sub_set_bus"
            ]

        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space, attr_to_keep = act_attr_to_keep)
        self.action_space = self._gym_env.action_space
        
    def setup_actions_all_actions(self):
        self._gym_env.action_space.close()
        act_attr_to_keep = [
            "change_bus", "change_line_status", "one_line_change", "one_line_set", "one_sub_change",
            "one_sub_set", "raise_alarm", "raise_alert", "set_bus", "set_line_status", "sub_change_bus",
            "sub_set_bus"
            ]
        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space, attr_to_keep= act_attr_to_keep)
        self.action_space = self._gym_env.action_space

    def setup_actions_discrete(self):
        self._gym_env.action_space.close()
        

        self._gym_env.action_space = DiscreteActSpace(self._g2op_env.action_space)
        self.action_space = self._gym_env.action_space
    def setup_actions_all_cont(self):
        self._gym_env.action_space.close()
        self._gym_env.action_space = gym_compat.BoxGymActSpace(self._g2op_env.action_space)
        self.action_space = Box(shape=self._gym_env.action_space.shape,
                                     low=np.array([0, 0, 0, -5, -10, -15],dtype=int),
                                     high=np.array([1, 1, 1, 5, 10, 15],dtype=int))
    

    def reset(self, seed=None, options=None):

        return self._gym_env.reset(seed=seed, options=options)

    def step(self, action):
        # for i in range(len(action)):
        #     action[i] = self.round_and_clip(action[i],self._gym_env.action_space.low[i],self._gym_env.action_space.high[i])
        next_obs = self._gym_env.step(action)
        

        return next_obs

    def render(self):
        return self._gym_env.render()
    

mda_env = Gym2OpEnv("mda")
mda_model = A2C.load("/mnt/c/Users/lukek/OneDrive/Desktop/Computer Science Honours/RL/RL-Assignment/src/actor_critic/actor_critic_models/model_MDA.zip")
mda_model.set_env(mda_env)



reward_log = "./MDA_rewards.txt"
episode_len_log = "./MDA_length.txt"
for i in range(500):
    done = False

    obs, info = mda_env.reset(options={"init ts": random.randint(60,1000)})
    ep_len = 0
    rewards = 0
    while done == False:
        
        action = mda_model.predict(obs)
        
        
        
        action = np.array(action[0])
        obs, reward, terminated, truncated, info = mda_env.step(action)
        if reward == -0.5:
            done = True        
        ep_len +=1
        rewards +=reward
        
    with open(reward_log, 'a') as firstfile:  
        firstfile.write(str(rewards) + '\n')
    firstfile.close()
    with open(episode_len_log, 'a') as secondfile:  
        secondfile.write(str(ep_len)+ '\n')
    secondfile.close()
    
   

