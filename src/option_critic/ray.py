import gymnasium as gym
from gymnasium.spaces import Box
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward

from lightsim2grid import LightSimBackend
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO


# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

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
        self._gym_env.observation_space.close()
        self._gym_env.observation_space = gym_compat.BoxGymObsSpace(self._g2op_env.observation_space)
        self.observation_space = Box(shape=self._gym_env.observation_space.shape,
                                     low=self._gym_env.observation_space.low,
                                     high=self._gym_env.observation_space.high)

    def setup_actions(self):
        self._gym_env.action_space.close()
        self._gym_env.action_space = gym_compat.BoxGymActSpace(self._g2op_env.action_space)
        self.action_space = Box(shape=self._gym_env.action_space.shape,
                                    low=self._gym_env.action_space.low,
                                    high=self._gym_env.action_space.high)

    def reset(self, seed=None):
        return self._gym_env.reset(seed=seed, options=None)

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        # TODO: Modify for your own required usage
        return self._gym_env.render()


def plot_comparison(random_rewards, ppo_rewards):
    """Plots the comparison of rewards between the Random and PPO agent."""
    plt.plot(random_rewards, label="Random Agent")
    plt.plot(ppo_rewards, label="PPO Agent")
    plt.title("Comparison of Total Rewards: Random vs PPO")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def run_random_agent(env, num_episodes, max_steps):
    """Run the random agent and collect rewards."""
    all_rewards = []

    for episode in range(num_episodes):
        print(f"Starting Random Agent Episode {episode + 1}")

        curr_step = 0
        curr_return = 0
        is_done = False

        obs, info = env.reset()

        while not is_done and curr_step < max_steps:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)

            curr_step += 1
            curr_return += reward
            is_done = terminated or truncated

        print(f"Random Agent Episode {episode + 1} finished with total reward = {curr_return}")
        all_rewards.append(curr_return)

    return all_rewards


def run_ppo_agent(env, model, num_episodes, max_steps):
    """Run the trained PPO agent and collect rewards."""
    all_rewards = []

    for episode in range(num_episodes):
        print(f"Starting PPO Agent Episode {episode + 1}")

        curr_step = 0
        curr_return = 0
        is_done = False

        obs, info = env.reset()

        while not is_done and curr_step < max_steps:
            action, _ = model.predict(obs)  # Use the trained PPO agent to select an action
            obs, reward, terminated, truncated, info = env.step(action)

            curr_step += 1
            curr_return += reward
            is_done = terminated or truncated

        print(f"PPO Agent Episode {episode + 1} finished with total reward = {curr_return}")
        all_rewards.append(curr_return)

    return all_rewards


def main():
    max_steps = 100
    num_episodes = 100  # Adjust the number of episodes
    total_timesteps = 20000  # Total timesteps for training the RL agent

    env = Gym2OpEnv()

    # Run the random agent and collect its rewards
    random_rewards = run_random_agent(env, num_episodes, max_steps)

    # Initialize the PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the PPO agent
    print(f"Training PPO agent for {total_timesteps} timesteps.")
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save("ppo_grid2op_model")

    # Run the PPO agent and collect its rewards
    ppo_rewards = run_ppo_agent(env, model, num_episodes, max_steps)

    # Compare the performance of both agents by plotting the rewards
    plot_comparison(random_rewards, ppo_rewards)


if __name__ == "__main__":
    main()