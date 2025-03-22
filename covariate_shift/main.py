import numpy as np
import argparse
import yaml
import os
from analysis import save_total_nll

def main(settings_file):
    with open(settings_file, 'r') as f:
        settings = yaml.safe_load(f)
    output_dir = settings['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    distributions = settings['experiment']['current_distribution']
    n_samples_per_env = settings['experiment']['n_samples_per_env']
    M = settings['experiment']['M']
    N = settings['experiment']['N']
    true_theta_E_list = settings['experiment']['true_theta_E_list']
    theta_j_possible = np.linspace(0, 1, M+3, endpoint=True)[1:-1]  # Possible mechanism parameters. In total M parameters
    ground_truth_k_list = [len(set(theta_E)) for theta_E in true_theta_E_list]  # Corresponding ground-truth k values (number of unique mechanisms)
    k_max_value = settings['experiment']['k_max_value']
    k_values = list(range(1, k_max_value + 1))
    seed = settings['experiment']['seed']
    method = settings['experiment']['method']
    save_total_nll(output_dir, distributions, n_samples_per_env, M, N, true_theta_E_list, ground_truth_k_list,
                                k_values, seed, theta_j_possible, method)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--settings_file', type=str, required=True, help='Path to the settings YAML file.')
    args = parser.parse_args()

    main(args.settings_file)