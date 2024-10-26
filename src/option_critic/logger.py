# Code adapted from : https://github.com/lweitkamp/option-critic-pytorch/
import logging
import os
import time
import numpy as np
import csv
from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self, logdir, run_name):
        self.log_name = logdir + '/' + run_name
        self.tf_writer = None
        self.start_time = time.time()
        self.n_eps = 0

        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.writer = SummaryWriter(self.log_name)

        # Set up logging to console and file
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_name + '/logger.log'),
            ],
            datefmt='%Y/%m/%d %I:%M:%S %p'
        )

        # Initialize CSV file for logging episode rewards and lengths
        self.eps_csv_file = self.log_name + '/episode_logs.csv'
        self.loss_csv_file = self.log_name + '/loss_logs.csv'
        with open(self.eps_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Total_Steps', 'Reward', 'Episode_Length', 'Epsilon'])

        with open(self.eps_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Total_Steps', 'Reward', 'Episode_Length', 'Epsilon'])

    def log_episode(self, steps, reward, option_lengths, ep_steps, epsilon):
        self.n_eps += 1
        logging.info(f"> ep {self.n_eps} done. total_steps={steps} | reward={reward} | episode_steps={ep_steps} "
                     f"| hours={(time.time()-self.start_time) / 60 / 60:.3f} | epsilon={epsilon:.3f}")
        self.writer.add_scalar(tag="episodic_rewards", scalar_value=reward, global_step=self.n_eps)
        self.writer.add_scalar(tag='episode_lengths', scalar_value=ep_steps, global_step=self.n_eps)

        # Log episode statistics to CSV
        with open(self.eps_csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.n_eps, steps, reward, ep_steps, epsilon])

        # Keep track of options statistics
        for option, lens in option_lengths.items():
            # Write statistics for each option
            self.writer.add_scalar(tag=f"option_{option}_avg_length", scalar_value=np.mean(lens) if len(lens) > 0 else 0, global_step=self.n_eps)
            self.writer.add_scalar(tag=f"option_{option}_active", scalar_value=sum(lens) / ep_steps, global_step=self.n_eps)

    def log_data(self, step, actor_loss, critic_loss, entropy, epsilon):
        if actor_loss:
            self.writer.add_scalar(tag="actor_loss", scalar_value=actor_loss.item(), global_step=step)
        if critic_loss:
            self.writer.add_scalar(tag="critic_loss", scalar_value=critic_loss.item(), global_step=step)
        self.writer.add_scalar(tag="policy_entropy", scalar_value=entropy, global_step=step)
        self.writer.add_scalar(tag="epsilon", scalar_value=epsilon, global_step=step)

# Example usage
if __name__ == "__main__":
    logger = Logger(logdir='runs/', run_name='test_model-test_env')
    steps = 200
    reward = 5
    option_lengths = {opt: np.random.randint(0, 5, size=(5)) for opt in range(5)}
    ep_steps = 50
    logger.log_episode(steps, reward, option_lengths, ep_steps, 1e-6)
