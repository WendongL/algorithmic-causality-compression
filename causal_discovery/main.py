import argparse
import yaml
import os
from analysis import save_total_nll, generate_candidates

def main(settings_file):
    with open(settings_file, 'r') as f:
        settings = yaml.safe_load(f)
    output_dir = settings['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    distributions = settings['experiment']['distributions']
    n_samples_per_env = settings['experiment']['n_samples_per_env']
    N = settings['experiment']['N']
    true_theta_E = settings['experiment']['true_theta_E']
    k_max_value = settings['experiment']['k_max_value']
    seed = settings['experiment']['seed']
    direction = settings['experiment']['direction']
    num_candidates = settings['experiment']['num_candidates']

    sigma2_X1_possible, sigma2_X2_possible, linear_coeff_possible = generate_candidates(
        true_theta_E,
        direction='X1->X2',
        num_candidates=num_candidates,
        method='grid'
    )

    tau2_X1_possible, tau2_X2_possible, anti_linear_coeff_possible = generate_candidates(
        true_theta_E,
        direction='X2->X1',
        num_candidates=num_candidates,
        method='grid'
    )
    save_total_nll(output_dir, distributions, n_samples_per_env, N, true_theta_E, k_max_value,
                seed, direction, sigma2_X1_possible, sigma2_X2_possible, linear_coeff_possible,
                tau2_X1_possible, tau2_X2_possible, anti_linear_coeff_possible)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments.')
    parser.add_argument('--settings_file', type=str, required=True, help='Path to the settings YAML file.')
    args = parser.parse_args()

    main(args.settings_file)