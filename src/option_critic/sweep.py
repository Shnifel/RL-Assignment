import argparse
from Assignment.model_2.env import Gym2OpEnv
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        env = Gym2OpEnv()

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            n_steps=config.n_steps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            n_epochs=config.n_epochs,
            verbose=2
        )

        model.learn(
            total_timesteps=config.total_timesteps,
            progress_bar=True,
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{wandb.run.id}",
                verbose=2,
            ),
        )

if __name__ == "__main__":
    # sweep configuration
    sweep_config = {
        'method': 'random',  # 'grid', 'random', or 'bayes'
        'metric': {'name': 'reward', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {'values': [1e-3, 3e-4, 1e-4]},
            'batch_size': {'values': [32, 64, 128]},
            'n_steps': {'values': [1024, 2048, 4096]},
            'gamma': {'values': [0.99, 0.98]},
            'gae_lambda': {'values': [0.95, 0.9]},
            'clip_range': {'values': [0.2, 0.1]},
            'ent_coef': {'values': [0.01, 0.001]},
            'n_epochs': {'values': [10, 20]},
            'total_timesteps': {'value': int(1e6)}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="PPO_Sweep")

    wandb.agent(sweep_id, function=train, count=10)  # can change count for the number of trials
