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
    stats_list = []
    for subfolder in os.listdir(dir):
        subfolder_path = os.path.join(dir, subfolder)
        
        if os.path.isdir(subfolder_path):
            csv_files = glob.glob(os.path.join(subfolder_path, "*.csv"))
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                all_results.append(df)
                run_names.append(subfolder)

                reward_max = df['Reward'].max()
                reward_min = df['Reward'].min()
                
                # Extract the last 10 episodes' rewards
                last_10_rewards = df['Reward'].tail(10)
                
                # Calculate Mean and Variance of the last 10 rewards
                last10_mean = last_10_rewards.mean()
                last10_variance = last_10_rewards.std()
                
                # Append the statistics to the stats_list
                stats_list.append({
                    'Folder': subfolder,
                    'Last10_Episode_Mean': last10_mean,
                    'Last10_Episode_Variance': last10_variance
                })

    return all_results, run_names, stats_list

def compile_reward_curves(all_results, run_names, results_dir, ftitle, fnames, fdescrip, save_name, colors, K = 100):
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    i = 0
    for df, run_name in zip(all_results, run_names):

        if run_name not in fnames:
            continue

        # Extract only up to 1 mil eps
        df_filtered = df[df['Total_Steps'] <= 1000000]
        
        # Apply smoothing
        df_filtered['Smoothed_Reward'] = df_filtered['Reward'].rolling(window=K, center=True, min_periods=1).mean()
        df_filtered['Reward_STD'] = df_filtered['Reward'].rolling(window=K,center=True, min_periods=1).std()
        df_filtered['Reward_STD'].fillna(0, inplace=True)
        
        plt.plot(df_filtered['Total_Steps'], df_filtered['Smoothed_Reward'], label=fdescrip[i], color = colors[i])
        plt.fill_between(
            df_filtered['Total_Steps'],
            df_filtered['Smoothed_Reward'] - df_filtered['Reward_STD'],
            df_filtered['Smoothed_Reward'] + df_filtered['Reward_STD'],
            alpha=0.2,
            color = colors[i]
        )
        i += 1
    plt.xlabel('Total Steps')
    plt.ylabel(f'Episode Rewards')
    plt.title(f'Training rewards over steps for {ftitle}\n(smoothed with window of {K})')
    if len(fdescrip) != 1:
        plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(results_dir, f'{save_name}.pdf')
    plt.savefig(plot_filename)
    plt.close()

if __name__ == "__main__":
    args = parser.parse_args()
    all_results, run_names, stats = extract_train_rewards(dir=args.results_dir)
    files = [["attention", "attention_4", "attention_6"],
             ["baseline_cont", "baseline_disc"],
             ["baseline_cont", "pruned_obs_space_baseline"],
             ["reward_shaping"],
             ["changed_q_net"],
             ["reparam_actions"]]
    
    file_descrips = [["Options = 2", "Options = 4", "Options = 6"],
                     ["Continuous action space", "Discrete action space"],
                     ["Original observation space", "Culled observation space"],
                     [""], [""], [""]]
    
    colors = [["red", "green", "blue"], ["orange", "blue"], ["orange", "blue"], ["dodgerblue"], ["lightsalmon"], ["blueviolet"]]
    
    ftitles = ["attention-based option critic", "baselines for different action encodings", "pruned observation spaces",
               "episode length reward shaping", "modified Q-Network architecture", "action reparameterisation"]
    
    save_names = ["attention", "baseline", "pruned_obs", "reward_shape", "modified_q", "action_reparam"]

    for file, file_descrip, ftitle, save_name, color in zip(files, file_descrips, ftitles, save_names, colors):
        compile_reward_curves(all_results, run_names, args.save_dir, ftitle, file, file_descrip, save_name, color, K = args.K)

    df_stats = pd.DataFrame(stats)
    
    
    output_dir = os.path.dirname(args.save_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    df_stats.to_csv(os.path.join(args.save_dir, "eval_stats.csv"), index=False)