import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import numpy as np


# Will auto select the latest run if None
log_dir = None
POLL_INTERVAL = 1
# Window size for the moving average. Set to 1 to disable smoothing.
SMOOTHING_WINDOW = 1
#log_dir = r"C:\School\NextGenDynamics\ChargeProject\logs\skrl\spiderbot\2025-10-21_10-22-17_ppo_torch"
base_log_dir = "logs/skrl/spiderbot"


# Parses tfevents into a DataFrame
def parse_tfevents_to_dataframe(acc):
    acc.Reload()  # Reloads the file to get the latest data
    
    tags = acc.Tags()['scalars']
    
    all_data = []
    for tag in tags:
        # Filter for only the reward-related tags at the source
        if tag.startswith('Info / Episode_Reward/'):
            events = acc.Scalars(tag)
            for event in events:
                all_data.append({
                    'tag': tag.replace('Info / Episode_Reward/', ''),
                    'step': event.step,
                    'value': event.value
                })
                
    if not all_data:
        return pd.DataFrame() # Return empty DataFrame if no reward data yet

    return pd.DataFrame(all_data)

# If log_dir is not set get the latest folder in logs/skrl/quadre
if log_dir is None:
    all_runs = glob.glob(os.path.join(base_log_dir, "*"))
    log_dir = max(all_runs, key=os.path.getmtime)
    print(f"No log_dir specified. Using the latest run: {log_dir}")

# Find the .tfevents file automatically
event_file_path = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))[0]


print(f"Reading data from: {event_file_path}")

# 2. Setup the plot
plt.ion() # Turn on interactive mode
fig, ax = plt.subplots(figsize=(16, 9))
lines = {} # Dictionary to store line objects, mapping tag_name to a line
event_acc = event_accumulator.EventAccumulator(event_file_path)

# Use a colormap to automatically assign different colors to new lines
colormap = plt.get_cmap('tab20')
colors = [colormap(i) for i in np.linspace(0, 1, 20)]
color_index = 0

print("Starting live plot. Press Ctrl+C to exit.")

# 3. Live update loop
try:
    while True:
        # Read all available scalar data
        reward_tags_df = parse_tfevents_to_dataframe(event_acc)

        if reward_tags_df.empty:
            print("No reward data found yet. Waiting...")
            plt.pause(POLL_INTERVAL)
            continue
            
        # Get all unique reward tags found so far
        unique_tags = reward_tags_df['tag'].unique()

        # Update or create a line for each reward component
        for tag_name in unique_tags:
            subset_df = reward_tags_df[reward_tags_df['tag'] == tag_name].copy()
            
            # Apply moving average for smoothing
            if SMOOTHING_WINDOW > 1:
                subset_df['value'] = subset_df['value'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

            clean_label = tag_name.replace('Episode_Reward/', '')
            
            # If this is a new tag, create a new line object
            if tag_name not in lines:
                line_color = colors[color_index % len(colors)]
                color_index += 1
                lines[tag_name] = ax.plot(
                    subset_df['step'], 
                    subset_df['value'], 
                    label=clean_label, 
                    alpha=0.9,
                    color=line_color
                )[0]
            # Otherwise, just update the data of the existing line
            else:
                lines[tag_name].set_data(subset_df['step'], subset_df['value'])

        # Customize the plot
        ax.set_title(f"Live Reward Components ({SMOOTHING_WINDOW}-Step Moving Average)", fontsize=18)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Smoothed Reward Value", fontsize=12)
        ax.legend(title="Reward Components", bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Rescale the axes to fit the new data
        ax.relim()
        ax.autoscale_view()
        
        # Redraw the canvas
        fig.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust for legend
        fig.canvas.draw()
        fig.canvas.flush_events()
        # print the most recent reward values to console
        """
        for tag_name, line in lines.items():
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            if len(xdata) > 0:
                print(f"{tag_name}: {ydata[-1]:.2f}")
        """

        plt.pause(POLL_INTERVAL)

except KeyboardInterrupt:
    print("\nPlotting stopped.")
finally:
    plt.ioff() # Turn off interactive mode
    # Optional: save the final plot
    # plt.savefig("final_reward_plot.png", dpi=300)
    print("Final plot window is open. You can close it manually.")
    plt.show() # Show the final plot and block until closed
    