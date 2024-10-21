from Assignment.model_2.env import Gym2OpEnv
from stable_baselines3 import PPO
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="PPO with Gym2Op and wandb integration")

    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for PPO")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for PPO")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Clipping range")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--total_timesteps", type=int, default=int(1e6), help="Total timesteps for training")

    parser.add_argument("--run_name", type=str, default="ppo_run", help="Wandb run name")    # wandb run name

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
# wandb
    run = wandb.init(
        project="PPO",
        config=vars(args), 
        name=args.run_name
    )
    
    env = Gym2OpEnv()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        n_epochs=args.n_epochs,
        verbose=2
    )

    # training the model
    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    run.finish()