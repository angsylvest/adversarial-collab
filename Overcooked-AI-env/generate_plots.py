import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_csv_files(csv_paths, labels, output_file=None):
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
    
    plt.title("Step vs Value")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

results_path = "results/"
csv_files = ['2024-11-20_14-50-28_experiment_asymmetric_advantages_alg_baseline_lazy_(False, 0.5)_adver_(True, 0.5)_include_[True, True, True, True]', 
'2024-11-21_17-14-29_experiment_asymmetric_advantages_alg_multiTravos_disc_True_adapt_False_UPDATED_lazy_(False, 0.5)_adver_(True, 0.5)_include_[True, True, True, True]']
for inx, file in enumerate(csv_files):
    csv_files[inx] = os.path.join(results_path, file)

labels = ['DDPG', 'Travos-DDPG']
env_type = "asymmetric_advantages_alg_baseline_lazy_(False, 0.5)_adver_(True, 0.5)_include_[True, True, True, True]"
plot_csv_files(csv_files, labels, output_file=f"{env_type}-plot.png")
