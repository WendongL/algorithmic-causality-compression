import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse
import matplotlib.colors as mcolors
import colorsys
import sys
current_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current_dir, '..'))
from utils import compute_penalty, load_config_from_yaml

def adjust_color_brightness(color, amount=1.0):
    """
    Adjust the brightness of the given color.
    amount > 1 makes the color lighter, amount < 1 makes it darker.
    """
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(*c)
    l = max(0, min(1, l * amount))
    return colorsys.hls_to_rgb(h, l, s)

def plot_total_coding_lengths(output_dir, dist_X1_given_E, dist_X2_given_X1, n_samples_per_env, seeds, k_max_value,
                              M, N, true_theta_E_list, plot_type='violin', with_penalty=True):
    # Aggregate total coding lengths across multiple files (seeds)
    k_values = list(range(1, k_max_value + 1))

    plt.figure(figsize=(12, 12))
    positions = np.array(k_values)

    colors = ['blue', 'green', 'red']

    for idx, true_theta_E in enumerate(true_theta_E_list):
        ground_truth_k = len(set(true_theta_E))
        true_theta_E = np.array(true_theta_E)

        print(f'\nProcessing n_samples_per_env={n_samples_per_env}, '
              f'Dist_X1_given_E={dist_X1_given_E}, Dist_X2_given_X1={dist_X2_given_X1}, '
              f'Ground Truth k={ground_truth_k}')
        print(f'Ground Truth theta_E: {true_theta_E}')

        # Initialize list to store total coding lengths for all seeds and k
        total_coding_lengths_seeds_k = {k: [] for k in k_values}

        for seed in seeds:
            file_path = f'{output_dir}/{dist_X1_given_E}_{dist_X2_given_X1}_samples{n_samples_per_env}_seed{seed}.pkl'
            if not os.path.exists(file_path):
                print(f'File {file_path} does not exist.')
                continue
            with open(file_path, 'rb') as f:
                pickle_data = pickle.load(f)
                theta_E_key = f'theta_E_{idx}'
                if theta_E_key in pickle_data:
                    total_coding_lengths_seeds_k_from_file = pickle_data[theta_E_key]
                    for k in k_values:
                        total_coding_lengths_k = total_coding_lengths_seeds_k_from_file.get(k, [])
                        if with_penalty:
                            penalty = compute_penalty(N, k, M)
                            print(f'Penalty for k={k}: {penalty}')
                            total_coding_lengths_k += penalty
                        total_coding_lengths_seeds_k[k].extend(total_coding_lengths_k)
                else:
                    print(f'No data for {theta_E_key} in file {file_path}')

        data_for_plot = [total_coding_lengths_seeds_k[k] for k in k_values]

        base_color = colors[idx % len(colors)]
        if plot_type == 'box':
            box_color = adjust_color_brightness(base_color, amount=1.2) 
            bp = plt.boxplot(data_for_plot, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor(box_color)
                patch.set_alpha(0.5)
                patch.set_edgecolor('black')
                patch.set_linewidth(1)
        elif plot_type == 'violin':
            violin_color = adjust_color_brightness(base_color, amount=1.2) 
            vp = plt.violinplot(data_for_plot, positions, widths=0.4, showmeans=False, showextrema=False)
            for i, body in enumerate(vp['bodies']):
                body.set_facecolor(violin_color)
                body.set_alpha(0.5)
                body.set_edgecolor('black')  # Add black contour
                body.set_linewidth(1)
                body.set_zorder(1)

                # Slightly shift overlapping violins
                x_shift = positions[i] + (idx - 1) * 0.05
                for path in body.get_paths():
                    path.vertices[:, 0] += (x_shift - positions[i])

        mean_values = [np.mean(lengths) if len(lengths) > 0 else np.nan for lengths in data_for_plot]
        curve_color = adjust_color_brightness(base_color, amount=0.8)  # Darker color for curves
        plt.plot(positions, mean_values, marker='o', color=curve_color,
                 label=f'Ground truth k={ground_truth_k},\n θ_E={true_theta_E}')

        # Highlight the minimal point for this curve with a yellow color and no annotation
        if np.any(~np.isnan(mean_values)):
            min_idx = np.nanargmin(mean_values)
            min_val = mean_values[min_idx]
            plt.scatter(positions[min_idx], min_val, marker='o', s=100, color='gold', zorder=3)

    plt.xticks(k_values)
    plt.xlabel('Number of Mechanisms (k)')
    plt.ylabel(r'FC complexity $(-\log P(X,E) + 2l(\alpha)+1$' if with_penalty else r'Negative Log-Likelihood $(-\log P(X,E))$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_filename = f'{output_dir}/plot_{dist_X1_given_E}_{dist_X2_given_X1}_samples{n_samples_per_env}_penalty{with_penalty}_{plot_type}.pdf'
    plt.savefig(output_filename, dpi=300)
    print(f'Figure saved as {output_filename}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot FC complexity from pickle files.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    config = load_config_from_yaml(args.config)

    # Extract values from the config
    output_dir = config.get('output_dir')
    dist_X1_given_E = config.get('experiment', {}).get('current_distribution', {}).get('X1', '')
    dist_X2_given_X1 = config.get('experiment', {}).get('current_distribution', {}).get('X2', '')
    n_samples_per_env = config.get('experiment', {}).get('n_samples_per_env', '')
    print(f'n_samples_per_env: {n_samples_per_env}')
    seeds = config.get('experiment', {}).get('seeds', [])
    k_max_value = config.get('experiment', {}).get('k_max_value', '')
    true_theta_E_list = config.get('experiment', {}).get('true_theta_E_list', [])
    M = config.get('experiment', {}).get('M', '')
    N = config.get('experiment', {}).get('N', '')
    k_values = list(range(1, k_max_value + 1))

    for with_penalty_flag in [False, True]:
        file_paths = [
            f'{output_dir}/{dist_X1_given_E}_{dist_X2_given_X1}_samples{n_samples_per_env}_penalty{with_penalty_flag}_seed{seed}.pkl'
            for seed in seeds
        ]
        print(f'file_paths: {file_paths}')
        if not file_paths:
            print(f'No files found matching file_paths {file_paths}')
            exit(1)
        print(f'Found {len(file_paths)} files.')

        plt.rcParams.update({'font.size': 22})
        plot_total_coding_lengths(output_dir, dist_X1_given_E, dist_X2_given_X1, n_samples_per_env, seeds, k_max_value,
                                  M, N, true_theta_E_list, plot_type='violin', with_penalty=with_penalty_flag)