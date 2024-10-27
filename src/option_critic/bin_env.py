import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend
import matplotlib.pyplot as plt
import numpy as np


# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(
            self,
            act_space_type = "disc"
    ):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE
        self.act_space_type = act_space_type

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
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
        self.setup_actions()

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

    def setup_observations(self):

        attributes_to_keep = [
            "a_ex", "a_or", "actual_dispatch",
            "delta_time",
            "gen_margin_down", "gen_margin_up", "gen_p", "gen_p_before_curtail", "gen_q",
             "gen_v","line_status", "load_p", "load_q", "load_v", "p_ex", "p_or", "prod_p", "prod_q", "prod_v",
            "q_ex", "q_or", "rho",  "target_dispatch", "thermal_limit", "v_ex", "v_or"
        ]
       
        # attributes_to_keep = [
        #     "load_p"
        #     ]

        self.bins = {
            "a_ex": np.linspace(0,1200,600),
            "a_or": np.linspace(0,1200,600),
            "actual_dispatch": np.linspace(-140,140,200),
            "delta_time": np.linspace(0,2088156454,100),
             "gen_margin_down": np.linspace(0,15,15),
             "gen_margin_up": np.linspace(0,15,15),
             "gen_p": np.linspace(-162,302,500),
             "gen_p_before_curtail": np.linspace(-162,302,500),
              "gen_q": np.linspace(-140,140,140),
              "gen_v": np.linspace(-140,140,140),
              "load_p": np.linspace(-140,140,140),
              "load_q": np.linspace(-140,140,140),
              "load_v": np.linspace(-140,140,140),
               "p_ex": np.linspace(-140,140,140),
               "p_or": np.linspace(-140,140,140),
               "prod_p": np.linspace(-162,302,500),
               "prod_q": np.linspace(-162,302,200),
                "prod_v": np.linspace(0,1000,100),
                "q_ex": np.linspace(-20,20,20),
                "q_or": np.linspace(-20,20,20),
                 "rho":np.linspace(0,5,10),
                 "target_dispatch": np.linspace(-140,140,560),
                 "thermal_limit": np.linspace(0,2000,1000),
                 "v_ex": np.linspace(0,200,100),
                  "v_or": np.linspace(0,200,100),
        }
        low, high = [], []
        self._gym_env.observation_space = gym_compat.BoxGymObsSpace(self._g2op_env.observation_space, attr_to_keep=attributes_to_keep)
        for attr in attributes_to_keep:
            if attr in self.bins:
                low.append(self.bins[attr][0])  
                high.append(self.bins[attr][-1])  # Max bin index
            else:
                low.append(self._gym_env.observation_space.low[attributes_to_keep.index(attr)])
                high.append(self._gym_env.observation_space.high[attributes_to_keep.index(attr)])
        
        self.observation_space = Box(shape=(len(attributes_to_keep),),
                                 low=np.array(low),
                                 high=np.array(high),
                                 dtype=int)
        
    def setup_actions(self):
        self._gym_env.action_space.close()
        if self.act_space_type == "cont":
            self._gym_env.action_space = gym_compat.BoxGymActSpace(self._g2op_env.action_space)
            self.action_space = Box(shape=self._gym_env.action_space.shape,
                                        low=self._gym_env.action_space.low,
                                        high=self._gym_env.action_space.high)
        elif self.act_space_type == "disc":
            act_attr_to_keep = [
           "set_bus", "set_line_status"
            ]
            self._gym_env.action_space = gym_compat.DiscreteActSpace(self._g2op_env.action_space, attr_to_keep=act_attr_to_keep)
            self.action_space = self._gym_env.action_space
        else:
            pass

    def bin_obs(self,obs):
       
        
        new_obs = obs[0]
        new_obs=np.array(new_obs)
        attributes_names = [
            "a_ex", "a_or", "actual_dispatch", "delta_time",
            "gen_margin_down", "gen_margin_up", "gen_p", "gen_p_before_curtail", 
            "gen_q", "gen_v", "load_p", "load_q", "load_v", "p_ex", "p_or", 
            "prod_p", "prod_q", "prod_v", "q_ex", "q_or", "rho", 
            "target_dispatch", "thermal_limit", "v_ex", "v_or"
        ]

        attributes = [20, 20, 6, 1, 6, 6, 6, 6, 6, 6, 20, 11, 11, 11, 20, 20, 
                    6, 6, 6, 20, 20, 20, 6, 20, 20, 20]
        
        count = 0
        for attr_name, attr_len in zip(attributes_names, attributes):
            if attr_name in self.bins:
                for i in range(attr_len):
                   
                    
                    new_obs[count] = np.digitize(obs[0][count], self.bins[attr_name]) - 1  
                    count += 1
            else:
                count += attr_len
       
        return new_obs

    def reset(self, seed=None):
        obs = self.bin_obs(self._gym_env.reset(seed=seed, options=None))
        _ = None
        return obs, _

    def step(self, action):
        other_stuff = self._gym_env.step(action)
        obs = self.bin_obs(other_stuff)
        
        done=False
        if other_stuff[1]==-0.5:
            done=True
        return obs, other_stuff[1], done, other_stuff[3], other_stuff[4]

    def render(self):
        return self._gym_env.render()
    
    


                 

# env = Gym2OpEnv()
# state=env.reset()
# action = env.action_space.sample()
# obs=env.step(action)

