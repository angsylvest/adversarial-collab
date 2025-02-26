import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_csv_files(csv_paths, labels, output_file=None, title=None):
    """
    Plots 'Step' vs 'Value' for multiple CSV files.

    Parameters:
    - csv_paths (list): List of paths to CSV files.
    - labels (list): List of labels for each CSV file line.
    - output_file (str): Optional, path to save the plot image.
    """
    if len(csv_paths) != len(labels):
        raise ValueError("The number of CSV paths must match the number of labels.")
    
    plt.figure(figsize=(10, 6))
    
    for path, label in zip(csv_paths, labels):
        if not os.path.exists(path):
            print(f"File {path} does not exist. Skipping.")
            continue
        # Read CSV and plot
        data = pd.read_csv(path)
        plt.plot(data['Step'], data['Value'], label=label)
    
    if not title: 
        plt.title("Asymmetric Advantages Train Performance (50% Adversarial)")
    else: 
        plt.title(f"{title}")

    plt.xlabel("Step")
    plt.xlim(0, 100000)
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def auto_plot_baselines(results_path):
    """
    Automatically plots baselines for each environment and saves the plots.

    Parameters:
    - results_path (str): Path to the main directory containing environments and baselines.
    """
    # Iterate through the environment directories
    for env_dir in os.listdir(results_path):
        env_path = os.path.join(results_path, env_dir)
        
        if os.path.isdir(env_path):
            print(f"Processing environment: {env_dir}")
            
            csv_paths = []
            labels = []

            baseline_path = env_path
            
            if os.path.isdir(baseline_path):
                print(f"  Processing env: {baseline_path}")
                
                # Iterate through CSV files in each baseline directory
                for csv_file in os.listdir(baseline_path):
                    if csv_file.endswith(".csv"):
                        file_path = os.path.join(baseline_path, csv_file)
                        csv_paths.append(file_path)
                        
                        # Assign label based on keywords in filename
                        if 'baseline' in csv_file.lower():
                            label = 'DDPG'  # Example: Map to 'DDPG' if 'baseline' is in filename
                        elif 'multitravos' in csv_file.lower():
                            label = "DDPG-MultiTravos"
                        else:
                            label = "DDPG-Travos" # Use the first part of the file name as label (or customize)
                        labels.append(label)
            
            if csv_paths:
                # Generate output filename based on environment
                output_file = os.path.join(results_path, f"{env_dir}_performance_plot.png")
                env_dirs = {"asym": "Asymmetric Advantages", "coord": "Coordination Ring", "cramp": "Cramped Room"}
                env_type = env_dirs[env_dir.split("_")[0]]
                if "lazy" in env_dir:
                   title = f"{env_type} Train Performance (0.5 Lazy)"
                else: 
                    title = f"{env_type} Train Performance (0.5 Adv)"
                # Plot for the current environment
                plot_csv_files(csv_paths, labels, output_file=output_file, title=title)

# Main directory containing all the environments and baselines
results_path = "evals"
auto_plot_baselines(results_path)
