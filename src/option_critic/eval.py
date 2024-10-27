import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import glob


parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--results-dir', default='./logs', help='Results directory')
parser.add_argument('--save-dir', default='./results', help='Results  save directory')
parser.add_argument('--K', default=100, type=int, help='Smoothing window length')


def extract_train_rewards(dir):

    all_results = []
    run_names = []
    for subfolder in os.listdir(dir):
        subfolder_path = os.path.join(dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            csv_files = glob.glob(os.path.join(subfolder_path, "*.csv"))
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                all_results.append(df)
                run_names.append(subfolder)

    return all_results, run_names

def compile_reward_curves(all_results, run_names, results_dir, K = 99):
    os.makedirs(results_dir, exist_ok=True)

    for df, run_name in zip(all_results, run_names):

        # Extract only up to 1 mil eps
        df_filtered = df #[df['Total_Steps'] <= 1000000]
        
        # Apply smoothing
        df_filtered['Smoothed_Reward'] = df_filtered['Reward'].rolling(window=K).mean()
        df_filtered['Reward_STD'] = df_filtered['Reward'].rolling(window=K, min_periods=1).std()
        df_filtered['Reward_STD'].fillna(0, inplace=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_filtered['Total_Steps'], df_filtered['Smoothed_Reward'], label='Smoothed Reward')
        plt.fill_between(
            df_filtered['Total_Steps'],
            df_filtered['Smoothed_Reward'] - df_filtered['Reward_STD'],
            df_filtered['Smoothed_Reward'] + df_filtered['Reward_STD'],
            color='blue',
            alpha=0.2,
            label='±1 Std Dev'
        )
        plt.xlabel('Total Steps')
        plt.ylabel(f'Training Rewards (smoothed with window of {K})')
        plt.title(f'Rewards Curve for {run_name}')
        plt.legend()
        plt.grid(True)
        plot_filename = os.path.join(results_dir, f'{run_name}.pdf')
        plt.savefig(plot_filename)
        plt.close()

if __name__ == "__main__":
    args = parser.parse_args()
    all_results, run_names = extract_train_rewards(dir=args.results_dir)
    compile_reward_curves(all_results, run_names, results_dir=args.save_dir, K = args.K)
