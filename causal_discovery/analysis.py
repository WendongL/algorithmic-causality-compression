import numpy as np
import itertools
import pickle
from scipy.stats import multivariate_normal
from tqdm import tqdm
import sys
from estimation import estimate_theta_E_hat
from assignment import assign_environments_to_mechanisms

def compute_total_nll(partition_sigma2_X1, partition_sigma2_X2, partition_linear_coeff, mechanisms, X_samples_all_env, N, direction):
    total_nll = 0
    theta_E_assigned = np.zeros((N, 3))
    for cluster_idx_X1, cluster_X1 in enumerate(partition_sigma2_X1):
        for env in cluster_X1:
            theta_E_assigned[env, 0] = mechanisms[0][cluster_idx_X1]
    for cluster_idx_X2, cluster_X2 in enumerate(partition_sigma2_X2):
        for env in cluster_X2:
            theta_E_assigned[env, 1] = mechanisms[1][cluster_idx_X2]
    for cluster_idx_linear_coeff, cluster_linear_coeff in enumerate(partition_linear_coeff):
        for env in cluster_linear_coeff:
            theta_E_assigned[env, 2] = mechanisms[2][cluster_idx_linear_coeff]

    for E in range(N):
        sigma2_X1_loc, sigma2_X2_loc, linear_coeff_loc = theta_E_assigned[E, :]
        if direction == 'X1->X2':
            Sigma_P_loc = np.array([
                [sigma2_X1_loc, linear_coeff_loc * sigma2_X1_loc],
                [linear_coeff_loc * sigma2_X1_loc, linear_coeff_loc ** 2 * sigma2_X1_loc + sigma2_X2_loc]
            ])
        elif direction == 'X2->X1':
            Sigma_P_loc = np.array([
                [linear_coeff_loc ** 2 * sigma2_X2_loc + sigma2_X1_loc, sigma2_X2_loc * linear_coeff_loc],
                [sigma2_X2_loc * linear_coeff_loc, sigma2_X2_loc]
            ])
        mvn = multivariate_normal(np.zeros(2), Sigma_P_loc)
        n_samples_per_env = X_samples_all_env.shape[1]
        log2_P_E = np.log2(N)
        for i in range(n_samples_per_env):
            X = X_samples_all_env[E, i, :]
            log2_P_X = np.log2(mvn.pdf(X))
            total_nll += -log2_P_X - log2_P_E
    return total_nll


def anti_causal_params(sigma2_X1, sigma2_X2, linear_coeff):
    tau2_X2 = sigma2_X1 * linear_coeff ** 2 + sigma2_X2
    anti_linear_coeff = (sigma2_X1 * linear_coeff) / tau2_X2
    tau2_X1 = sigma2_X1 - tau2_X2 * (anti_linear_coeff ** 2)
    return tau2_X1, tau2_X2, anti_linear_coeff


def extract_param_values(true_theta_E, direction):
    true_theta_E = np.array(true_theta_E)
    if direction == 'X1->X2':
        sigma2_X1_vals = true_theta_E[:, 0]
        sigma2_X2_vals = true_theta_E[:, 1]
        linear_coeff_vals = true_theta_E[:, 2]
    elif direction == 'X2->X1':
        anti_params = [anti_causal_params(*theta) for theta in true_theta_E]
        anti_params = np.array(anti_params)
        sigma2_X1_vals = anti_params[:, 0]  # tau2_X1
        sigma2_X2_vals = anti_params[:, 1]  # tau2_X2
        linear_coeff_vals = anti_params[:, 2]  # anti_linear_coeff
    else:
        raise ValueError("Unsupported direction")
    return sigma2_X1_vals, sigma2_X2_vals, linear_coeff_vals


def extract_param_ranges(true_theta_E, direction, margin_ratio=0.1):
    sigma2_X1_vals, sigma2_X2_vals, linear_coeff_vals = extract_param_values(true_theta_E, direction)

    def add_margin(vals, positive=False):
        vmin, vmax = np.min(vals), np.max(vals)
        vrange = vmax - vmin
        if positive:
            return (max(vmin - margin_ratio * vrange, 0), vmax + margin_ratio * vrange)
        else:
            return (vmin - margin_ratio * vrange, vmax + margin_ratio * vrange)

    param_ranges = {
        'sigma2_X1': add_margin(sigma2_X1_vals, positive=True),
        'sigma2_X2': add_margin(sigma2_X2_vals, positive=True),
        'linear_coeff': add_margin(linear_coeff_vals)
    }

    return param_ranges


def generate_candidates(true_theta_E, direction, num_candidates, method='random',
                        margin_ratio=0.1, random_state=None):
    rng = np.random.default_rng(random_state)
    param_ranges = extract_param_ranges(true_theta_E, direction, margin_ratio=margin_ratio)
    sigma2_X1_vals, sigma2_X2_vals, linear_coeff_vals = extract_param_values(true_theta_E, direction)

    def generate_values(p_range, ground_truth_values):
        p_min, p_max = p_range
        unique_gt = np.unique(ground_truth_values)
        gt_count = len(unique_gt)
        extra_count = max(0, num_candidates - gt_count)

        if extra_count > 0:
            if method == 'random':
                extra_vals = rng.uniform(p_min, p_max, size=extra_count)
            elif method == 'grid':
                extra_vals = np.linspace(p_min, p_max, extra_count)
            else:
                raise ValueError("Unsupported generation method")
            combined = np.concatenate([unique_gt, extra_vals])
        else:
            combined = unique_gt

        combined = np.unique(combined)
        shortfall = num_candidates - len(combined)
        if shortfall > 0:
            if method == 'random':
                top_up = rng.uniform(p_min, p_max, size=shortfall)
            else:
                top_up = np.linspace(p_min, p_max, shortfall)
            combined = np.unique(np.concatenate([combined, top_up]))

        return np.sort(combined)

    sigma2_X1_candidates = generate_values(param_ranges['sigma2_X1'], sigma2_X1_vals)
    sigma2_X2_candidates = generate_values(param_ranges['sigma2_X2'], sigma2_X2_vals)
    linear_coeff_candidates = generate_values(param_ranges['linear_coeff'], linear_coeff_vals)

    return list(sigma2_X1_candidates), list(sigma2_X2_candidates), list(linear_coeff_candidates)



def save_total_nll(output_dir, distributions, n_samples_per_env, N, true_theta_E, k_values_max,
                   seed, direction, sigma2_X1_possible, sigma2_X2_possible, linear_coeff_possible,
                   tau2_X1_possible, tau2_X2_possible, anti_linear_coeff_possible):

    np.random.seed(seed)
    results = {}

    true_theta_E = np.array(true_theta_E)
    print(f"Processing N={N}, Direction={direction}, true_theta_E={true_theta_E}")
    sys.stdout.flush()
    from data_generation import generate_data
    X_samples_all_env = generate_data(distributions, N, n_samples_per_env, true_theta_E, seed)

    if direction == 'X1->X2':
        direction_name = 'causal'
        zeta2_X1_possible = sigma2_X1_possible
        zeta2_X2_possible = sigma2_X2_possible
        coeff_possible = linear_coeff_possible
    elif direction == 'X2->X1':
        direction_name = 'anticausal'
        zeta2_X1_possible = tau2_X1_possible
        zeta2_X2_possible = tau2_X2_possible
        coeff_possible = anti_linear_coeff_possible
    else:
        raise ValueError('Unsupported direction')

    num_candidates = len(zeta2_X1_possible)
    theta_E_hat = estimate_theta_E_hat(X_samples_all_env, N, distributions, direction)
    direction_results = {}

    # If any parameter dimension has candidates >= N, directly assign best mechanisms


    results[direction] = direction_results

    # Proceed with original exhaustive search
    for zeta2_X1_count in tqdm(range(1, min(k_values_max + 1, len(zeta2_X1_possible) + 1, N+1))):

        for zeta2_X2_count in tqdm(range(1, min(k_values_max + 1 - zeta2_X1_count, len(zeta2_X2_possible) + 1, N+1))):
            sys.stdout.flush()
            for coeff_count in range(1, min(k_values_max + 1 - zeta2_X1_count - zeta2_X2_count, len(coeff_possible) + 1, N+1)):

                # if (zeta2_X1_count >= N) and (zeta2_X2_count >= N) and (coeff_count >= N):
                #     # Direct assignment without exhaustive search
                #     partition_zeta2_X1, partition_zeta2_X2, partition_linear_coeff, kl_value_all_env = direct_assign_environments_to_mechanisms(
                #         theta_E_hat, zeta2_X1_possible, zeta2_X2_possible, coeff_possible, direction
                #     )
                #
                #     best_nll_all_env = compute_total_nll(
                #         partition_zeta2_X1, partition_zeta2_X2, partition_linear_coeff,
                #         (zeta2_X1_possible, zeta2_X2_possible, coeff_possible),
                #         X_samples_all_env, N, direction
                #     )
                #
                #     direction_results[(len(zeta2_X1_possible), len(zeta2_X2_possible), len(coeff_possible))] = {
                #         'best_nll_all_env': best_nll_all_env,
                #         'best_partition_zeta2_X1': partition_zeta2_X1,
                #         'best_partition_zeta2_X2': partition_zeta2_X2,
                #         'best_partition_linear_coeff': partition_linear_coeff,
                #         'best_mechanisms': (zeta2_X1_possible, zeta2_X2_possible, coeff_possible),
                #         'best_kl_given_k_vector': kl_value_all_env
                #     }
                # else:
                    all_possible_combi_zeta2_X1 = set(itertools.combinations(zeta2_X1_possible, zeta2_X1_count))
                    all_possible_combi_zeta2_X2 = set(itertools.combinations(zeta2_X2_possible, zeta2_X2_count))
                    all_possible_combi_linear_coeff = set(itertools.combinations(coeff_possible, coeff_count))
                    all_mechanisms_combinations = list(itertools.product(
                        all_possible_combi_zeta2_X1,
                        all_possible_combi_zeta2_X2,
                        all_possible_combi_linear_coeff
                    ))
                    best_kl_given_k_vector = float('inf')
                    for mechanisms in all_mechanisms_combinations:
                        partition_zeta2_X1, partition_zeta2_X2, partition_linear_coeff, kl_value_all_env = assign_environments_to_mechanisms(
                            theta_E_hat, mechanisms, direction)
                        if kl_value_all_env < best_kl_given_k_vector:
                            best_kl_given_k_vector = kl_value_all_env
                            best_partition_zeta2_X1 = partition_zeta2_X1
                            best_partition_zeta2_X2 = partition_zeta2_X2
                            best_partition_linear_coeff = partition_linear_coeff
                            best_mechanisms = mechanisms
                    best_nll_all_env = compute_total_nll(best_partition_zeta2_X1, best_partition_zeta2_X2, best_partition_linear_coeff,
                                                         best_mechanisms, X_samples_all_env, N, direction)

                    direction_results[(zeta2_X1_count, zeta2_X2_count, coeff_count)] = {
                        'best_nll_all_env': best_nll_all_env,
                        'best_partition_zeta2_X1': best_partition_zeta2_X1,
                        'best_partition_zeta2_X2': best_partition_zeta2_X2,
                        'best_partition_linear_coeff': best_partition_linear_coeff,
                        'best_mechanisms': best_mechanisms,
                        'best_kl_given_k_vector': best_kl_given_k_vector
                    }

            results[direction] = direction_results

    filename = f'{output_dir}/cd_distributions{distributions}_samples{n_samples_per_env}_seed{seed}_direction{direction_name}_candidates{num_candidates}_kmax{str(k_values_max)}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved total coding lengths to {filename}')
