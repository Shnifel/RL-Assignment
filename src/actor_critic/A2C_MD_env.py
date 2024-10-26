import gymnasium as gym
from gymnasium.spaces.box import Box
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward, BridgeReward
from grid2op.gym_compat import DiscreteActSpace
from grid2op.gym_compat import MultiDiscreteActSpace
from lightsim2grid import LightSimBackend
import matplotlib.pyplot as plt
import numpy as np
import joblib


# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(
            self
    ):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward 

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
        #cr.addReward("BridgeReward", BridgeReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########
        self.original_observation_space = self._g2op_env.observation_space
        self._gym_env = gym_compat.GymEnv(self._g2op_env)
        self.before_change = self._gym_env.observation_space
        self.setup_observations()
        self.setup_actions()

        # self.observation_space = self._gym_env.observation_space
        # self.action_space = self._gym_env.action_space

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
        # I think storage attributes can be removed
        #Removed: "was_alarm_used_after_game_over", "was_alert_used_after_attack",
        # "time_before_cooldown_line", "time_before_cooldown_sub", "time_next_maintenance",
        #     "time_since_last_alarm", "time_since_last_alert", "time_since_last_attack",
        #     "timestep_overflow",
        # "storage_charge", "storage_power", "storage_power_target",
        #     "storage_theta",
        #  "duration_next_maintenance", "gen_theta" "load_theta" , "theta_or" "theta_ex",
        # "total_number_of_alert",

        # attributes_to_keep = [
        #     "a_ex", "a_or", "active_alert", "actual_dispatch", "alert_duration",
        #     "attack_under_alert", "attention_budget", "current_step", "curtailment",
        #     "curtailment_limit", "curtailment_limit_effective", "curtailment_limit_mw",
        #     "curtailment_mw", "day", "day_of_week", "delta_time",
        #     "gen_margin_down", "gen_margin_up", "gen_p", "gen_p_before_curtail", "gen_q",
        #      "gen_v", "hour_of_day", "is_alarm_illegal", "last_alarm",
        #     "line_status", "load_p", "load_q", "load_v", "max_step",
        #     "minute_of_hour", "month", "p_ex", "p_or", "prod_p", "prod_q", "prod_v",
        #     "q_ex", "q_or", "rho",  "target_dispatch", "thermal_limit", 
        #      "topo_vect",  "v_ex", "v_or", "year"
        # ]
        
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = gym_compat.BoxGymObsSpace(self._g2op_env.observation_space, 
                                                                    attr_to_keep=attributes_to_keep)
        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                     low=self._gym_env.observation_space.low,
                                     high=self._gym_env.observation_space.high)
    


    def setup_actions(self):
        self._gym_env.action_space.close()
        
        
        act_attr_to_keep = [
            "change_bus", "change_line_status", "one_line_change", "one_line_set", "one_sub_change",
            "one_sub_set", "raise_alarm", "raise_alert", "set_bus", "set_line_status", "sub_change_bus",
            "sub_set_bus"
            ]
        # 3x compute improvement if anything with _bus is removed and minimal performance reduced
        # Can reduce to this with minimal loss and minimal performance increase
        # act_attr_to_keep = [
        #    "set_line_status", "set_bus","sub_set_bus" 
        #     ]
        # #Removed:    , change_bus, set_bus, change_line_status, "one_line_set",
        # Attributes that can do it alone : "set_line_status"(100 it/s) , "change_bus"(54 it/s), "set_bus"(53 it/s), "sub_change_bus"(100 it/s), "sub_set_bus" (100 it/s)
        # act_attr_to_keep = [
        #      "set_line_status"
        #     ]
        self._gym_env.action_space = MultiDiscreteActSpace(self._g2op_env.action_space, attr_to_keep = act_attr_to_keep)
        self.action_space = self._gym_env.action_space
        
        
        # self._gym_env.action_space = gym_compat.BoxGymActSpace(self._g2op_env.action_space)
        # self.action_space = Box(shape=self._gym_env.action_space.shape,
        #                             low=np.array([0, 0, 0, -5, -10, -15],dtype=int),
        #                             high=np.array([1, 1, 1, 5, 10, 15],dtype=int))
 
    def reset(self, seed=None, options=None):

        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        # for i in range(len(action)):
        #     action[i] = self.round_and_clip(action[i],self._gym_env.action_space.low[i],self._gym_env.action_space.high[i])
        
        return self._gym_env.step(action)
    
    
    
    def round_and_clip(self,action,low,high):
        rounded_action = np.round(action)
        clipped_action = np.clip(rounded_action, low, high)
        return clipped_action

    def render(self):
        return self._gym_env.render()
    
    def get_obs(self):
        
        return self._gym_env.observation_space

def main():
    # Random agent interacting in environment #

    max_steps = 100

    env = Gym2OpEnv()

    plt.imshow(env.render())
    plt.show()
    
    # print("#####################")
    # print("# OBSERVATION SPACE #")
    # print("#####################")
    # print(env.observation_space)
    # print("#####################\n")

    # print("#####################")
    # print("#   ACTION SPACE    #")
    # print("#####################")
    # print(env.action_space.sample())
    # print("#####################\n\n")

    curr_step = 0
    curr_return = 0

    is_done = False
    obs, info = env.reset()
    # print(f"step = {curr_step} (reset):")
    # print(f"\t obs = {obs}")
    # print(f"\t info = {info}\n\n")

    while not is_done and curr_step < max_steps:
        action = env.action_space.sample()
        print(type(action))
        obs = env.step(action)
        
        if curr_step < 2:
            print("Taken Action: ", action)
            
        curr_step += 1
        # curr_return += reward
        # is_done = done
        # print(f"step = {curr_step}: ")
        # print(f"\t obs = {obs}")
        # print(f"\t reward = {reward}")
        # print(f"\t terminated = {terminated}")
        # print(f"\t truncated = {truncated}")
        # print(f"\t info = {info}")

        # Some actions are invalid (see: https://grid2op.readthedocs.io/en/latest/action.html#illegal-vs-ambiguous)
        # Invalid actions are replaced with 'do nothing' action
        # is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
        # print(f"\t is action valid = {is_action_valid}")
        # if not is_action_valid:
        #     print(f"\t\t reason = {info['exception']}")
        # print("\n")

    print("###########")
    print("# SUMMARY #")
    print("###########")
    print(f"return = {curr_return}")
    print(f"total steps = {curr_step}")
    print("###########")


if __name__ == "__main__":
    main()
