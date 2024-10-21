import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# Specify the path to the event file
event_file = '/mnt/d/RL Assignment/RL-Labs-1/tensorboard_logs/MDA all actions and obs/events.out.tfevents.1729279684.DESKTOP-GJ8F1NB.8892.0'

# Load the event accumulator
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()  # Load the event data

# Get all available tags (e.g., scalars like 'loss', 'accuracy', etc.)
tags = ea.Tags()['scalars']

# Extract data for each tag and convert it to a pandas DataFrame
data_frames = {}
for tag in tags:
    events = ea.Scalars(tag)
    # Create a DataFrame with the step, value, and wall time
    df = pd.DataFrame([(e.step, e.value, e.wall_time) for e in events],
                      columns=['step', 'value', 'wall_time'])
    data_frames[tag] = df

    # Save each tag's data to a separate CSV file (optional)
    output_csv = f"{tag.replace('/', '_')}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Data for tag '{tag}' saved to {output_csv}")

# Optionally, if you want to combine all tags into a single DataFrame
combined_df = pd.concat(data_frames, axis=1)
combined_csv = "combined_tensorboard_data.csv"
combined_df.to_csv(combined_csv, index=False)
print(f"All data combined and saved to {combined_csv}")
