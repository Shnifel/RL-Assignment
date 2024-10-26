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
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = gym_compat.BoxGymObsSpace(self._g2op_env.observation_space, attr_to_keep=attributes_to_keep)
        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                     low=self._gym_env.observation_space.low,
                                     high=self._gym_env.observation_space.high)
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

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        return self._gym_env.render()

