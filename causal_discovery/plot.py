import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import os
import sys
current_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current_dir, '..'))
from utils import load_config_from_yaml, compute_penalty

def adjust_color_brightness(color, amount=1.0):
    """
    Adjust the brightness of the given color.
    amount >1 makes the color lighter, amount <1 makes it darker.
    """
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    h, l, s = colorsys.rgb_to_hls(*c)
    l = max(0, min(1, l * amount))
    return colorsys.hls_to_rgb(h, l, s)


def aggregate_data_from_files(files, output_dir, M_sigma2_X1, M_sigma2_X2, M_linear_coeff, N, with_penalty):
    """
    Given a set of files, all for the same direction, aggregate minimal coding length data by k.

    Returns: dict where keys are k and values are lists of minimal coding lengths across all files.
    """
    minimal_data_for_all_files = []

    for file_idx, results_file in enumerate(files):
        with open(os.path.join(output_dir, results_file), 'rb') as f:
            results = pickle.load(f)

        minimal_data_for_file = {}

        for direction, direction_dict in results.items():
            for (sigma2_X1_count, sigma2_X2_count, linear_coeff_count), vals in direction_dict.items():
                k = sigma2_X1_count + sigma2_X2_count + linear_coeff_count
                nll = vals['best_nll_all_env']
                if with_penalty:
                    p1 = compute_penalty(N, sigma2_X1_count, M_sigma2_X1)
                    p2 = compute_penalty(N, sigma2_X2_count, M_sigma2_X2)
                    p3 = compute_penalty(N, linear_coeff_count, M_linear_coeff)
                    p = p1 + p2 + p3
                else:
                    p = 0
                total_coding_length = nll + p

                # Track minimal value for this k in this file
                if k not in minimal_data_for_file:
                    minimal_data_for_file[k] = total_coding_length
                else:
                    if total_coding_length < minimal_data_for_file[k]:
                        minimal_data_for_file[k] = total_coding_length

        minimal_data_for_all_files.append(minimal_data_for_file)

    # Aggregate across files
    minimal_data = {}
    for data_for_file in minimal_data_for_all_files:
        for k, val in data_for_file.items():
            if k not in minimal_data:
                minimal_data[k] = []
            minimal_data[k].append(val)

    return minimal_data


def plot_combined_directions(output_dir, minimal_data_dict, with_penalty):
    """
    Given a dictionary like:
      minimal_data_dict = {
        'Causal': {k: [values across seeds], ...},
        'Anti-causal': {k: [values across seeds], ...}
      }
    plot them on one figure using violin plots and mean values.
    """

    directions = list(minimal_data_dict.keys())

    # Get all unique k values
    all_k_values = set()
    for direction in directions:
        all_k_values.update(minimal_data_dict[direction].keys())
    all_k_values = sorted(all_k_values)

    # Compute means
    direction_means = {}
    for direction in directions:
        direction_means[direction] = {k: np.mean(minimal_data_dict[direction][k]) for k in minimal_data_dict[direction]}

    # Find global minimal mean point
    global_min_val = None
    global_min_direction = None
    global_min_k = None
    for direction in directions:
        for k, mean_val in direction_means[direction].items():
            if (global_min_val is None) or (mean_val < global_min_val):
                global_min_val = mean_val
                global_min_direction = direction
                global_min_k = k

    # Prepare the figure
    plt.figure(figsize=(10, 6))
    base_colors = ['blue', 'red', 'green', 'purple', 'orange']
    direction_colors = {d: base_colors[i % len(base_colors)] for i, d in enumerate(directions)}

    # For the violin plot, we place each direction side-by-side at each k.
    width = 0.3
    for d_idx, direction in enumerate(directions):
        data_to_plot = []
        positions = []
        for k in all_k_values:
            if k in minimal_data_dict[direction]:
                data_to_plot.append(minimal_data_dict[direction][k])
                # Position offset
                positions.append(k + (d_idx - (len(directions) - 1)/2)*width)

        parts = plt.violinplot(data_to_plot, positions=positions, showmeans=False, showextrema=True, showmedians=True)
        curve_color = adjust_color_brightness(direction_colors[direction], amount=0.8)
        for pc in parts['bodies']:

            pc.set_facecolor(curve_color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

    # Plot mean values
    for d_idx, direction in enumerate(directions):
        mean_ys = []
        mean_xs = []
        for k in all_k_values:
            if k in direction_means[direction]:
                mean_ys.append(direction_means[direction][k])
                mean_xs.append(k + (d_idx - (len(directions)-1)/2)*width)
        plt.plot(mean_xs, mean_ys, marker='o', color=direction_colors[direction], label=direction)
        print("Direction:", direction)
        print("Mean Xs:", [round(x) for x in mean_xs])
        print("Mean YS:", [float(round(y, 1)) for y in mean_ys])

    # Highlight global minimal mean point
    d_min_idx = directions.index(global_min_direction)
    global_min_x = global_min_k + (d_min_idx - (len(directions)-1)/2)*width
    plt.plot(global_min_x, global_min_val, marker='o', color='gold', markersize=12,
             markeredgecolor='black', markeredgewidth=2)

    plt.xticks(all_k_values)
    plt.xlabel('Total Number of Mechanisms (k)')
    if with_penalty:
        plt.ylabel(r'FC complexity $(-\log P(X,E) + 2l(\alpha)+1$')
    else:
        plt.ylabel(r'Negative Log-Likelihood ($-\log P(X,E)$)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_filename = os.path.join(output_dir, f'cd_distributions{distributions}_samples{n_samples_per_env}_candidates{num_candidates}_kmax{str(k_max_value)}_with_penalty{with_penalty}_violin_combined.pdf')
    plt.savefig(output_filename, dpi=300)
    print(f'Figure saved as {output_filename}')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot FC complexity from pickle files.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()
    config = load_config_from_yaml(args.config)

    output_dir = config.get('output_dir')
    n_samples_per_env = config.get('experiment', {}).get('n_samples_per_env', '')
    seeds = config.get('experiment', {}).get('seeds', [])
    k_max_value = config.get('experiment', {}).get('k_max_value', '')
    N = config.get('experiment', {}).get('N', '')
    num_candidates = config.get('experiment', {}).get('num_candidates', '')
    distributions = config.get('experiment', {}).get('distributions', [])
    results_files_causal = [
        f'cd_distributions{distributions}_samples{n_samples_per_env}_seed{seed}_directioncausal_candidates{num_candidates}_kmax{str(k_max_value)}.pkl'
        for seed in seeds
    ]
    results_files_anticausal = [
        f'cd_distributions{distributions}_samples{n_samples_per_env}_seed{seed}_directionanticausal_candidates{num_candidates}_kmax{str(k_max_value)}.pkl'
        for seed in seeds
    ]
    for with_penalty in [True, False]:
        # Aggregate data for each direction
        minimal_data_causal = aggregate_data_from_files(results_files_causal, output_dir,
                                                        M_sigma2_X1=num_candidates, M_sigma2_X2=num_candidates,
                                                        M_linear_coeff=num_candidates, N=N, with_penalty=with_penalty)

        minimal_data_anticausal = aggregate_data_from_files(results_files_anticausal, output_dir,
                                                            M_sigma2_X1=num_candidates, M_sigma2_X2=num_candidates,
                                                            M_linear_coeff=num_candidates, N=N, with_penalty=with_penalty)


        minimal_data_dict = {
            'Causal': minimal_data_causal,
            'Anti-causal': minimal_data_anticausal
        }
        plt.rcParams.update({'font.size': 15})
        plot_combined_directions(output_dir, minimal_data_dict, with_penalty=with_penalty)