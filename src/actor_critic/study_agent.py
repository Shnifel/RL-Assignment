import os
import sys
import grid2op
import copy
import numpy as np
import shutil
import plotly.graph_objects as go
from grid2op.Parameters import Parameters
from tqdm import tqdm
from grid2op.Agent import PowerLineSwitch
from grid2op.Reward import L2RPNReward
from grid2op.Runner import Runner
from lightsim2grid import LightSimBackend
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward


path_agents = "/mnt/c/Users/lukek/OneDrive/Desktop/Computer Science Honours/RL/RL-Labs/Assignment/model_1/models/set_line_status_Best_model.zip"
bk_cls = LightSimBackend()


action_class = PlayableAction
observation_class = CompleteObservation
reward_class = CombinedScaledReward 

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
p = Parameters()
p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep
env_name = "l2rpn_case14_sandbox"
env = grid2op.make(env_name, backend=bk_cls, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p)

cr = env.get_reward_instance()
cr.addReward("N1", N1Reward(), 1.0)
cr.addReward("L2RPN", L2RPNReward(), 1.0)
# reward = N1 + L2RPN
cr.initialize(env)

shutil.rmtree(os.path.abspath(path_agents), ignore_errors=True)

runner = Runner(**env.get_params_for_runner(),
                agentClass=PowerLineSwitch
                )

res = runner.run(path_save=path_agents,
                 nb_episode=2, 
                 max_iter=10000,
                 nb_process=2,
                 pbar=tqdm)

print("The results for the evaluated agent are:")
for _, chron_id, cum_reward, nb_time_step, max_ts in res:
    msg_tmp = "\tFor chronics with id {}\n".format(chron_id)
    msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
    msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
    print(msg_tmp)
