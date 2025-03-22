import itertools
import pickle

import numpy as np
import sys
from assignment import assign_environments_to_mechanisms
from distributions import adjusted_log_poisson_pmf, adjusted_log_geometric_pmf, adjusted_log_discretized_gaussian_pmf, log_binomial_pmf
from estimation import estimate_theta_E_hat
from scipy.special import comb


def compute_total_nll(partition, mechanisms, theta_j_possible, E_samples, X1_samples, X2_samples,
                    N, k, M, dist_X1_given_E, dist_X2_given_X1, sigma2_X1=None, sigma2_X2=None):
    """
    Computes the total coding length given partitions, mechanisms, samples, and parameters.

    Parameters
    ----------
    partition : list of lists
        Environments assigned to each cluster.
    mechanisms : array-like
        Mechanism parameters for each cluster.
    theta_j_possible : list or array
        Possible values for theta_j used in computing P(X2 | X1).
    E_samples : array-like
        Samples of the environment variable.
    X1_samples : array-like
        Samples of the first output variable.
    X2_samples : array-like
        Samples of the second output variable.
    N : int
        Number of environments.
    k : int
        Number of clusters.
    M : int
        Number of possible mechanism parameters.
    dist_X1_given_E : str
        Distribution type for X1 given E ('bernoulli', 'poisson', 'geometric', 'gaussian').
    dist_X2_given_X1 : str
        Distribution type for X2 given X1 ('bernoulli', 'poisson', 'geometric', 'gaussian').
    sigma2_X1 : float, optional
        Variance of X1 (required for Gaussian distribution).
    sigma2_X2 : float, optional
        Variance of X2 given X1 (required for Gaussian distribution).

    Returns
    -------
    total_coding_length : float
        The total coding length for the given data and parameters.
    """
    theta_E_assigned = np.zeros(N)
    for cluster_idx, cluster in enumerate(partition):
        for env in cluster:
            theta_E_assigned[env] = mechanisms[cluster_idx]

    n_samples = len(E_samples)
    log_P_X = 0
    for i in range(n_samples):
        E = E_samples[i]
        X1 = X1_samples[i]
        X2 = X2_samples[i]
        theta = theta_E_assigned[E]

        P_E = 1 / N  # Uniform distribution over environments
        log_P_E_base2 = np.log2(N)

        # Compute P(X1 | E)
        if dist_X1_given_E == 'bernoulli':
            P_X1_given_E = theta if X1 == 1 else (1 - theta)
            log_P_X1_given_E_base2 = -np.log2(P_X1_given_E)
        elif dist_X1_given_E == 'poisson':
            lambda_E = theta * (M+2)
            log_P_X1_given_E = adjusted_log_poisson_pmf(X1, lambda_E, M)
            log_P_X1_given_E_base2 = -log_P_X1_given_E / np.log(2)
        elif dist_X1_given_E == 'geometric':
            p = theta
            log_P_X1_given_E = adjusted_log_geometric_pmf(X1, p, M)
            log_P_X1_given_E_base2 = -log_P_X1_given_E / np.log(2)
        elif dist_X1_given_E == 'gaussian':
            if sigma2_X1 is None:
                raise ValueError('sigma2_X1 must be provided for Gaussian distribution')
            sigma_X1 = np.sqrt(sigma2_X1)
            mu = theta * (M+2)
            log_P_X1_given_E = adjusted_log_discretized_gaussian_pmf(X1, mu, sigma_X1, M)
            log_P_X1_given_E_base2 = -log_P_X1_given_E / np.log(2)
        elif dist_X1_given_E == 'binomial':
            p = theta
            log_P_X1_given_E = log_binomial_pmf(X1, M+2, p)
            log_P_X1_given_E_base2 = -log_P_X1_given_E / np.log(2)
        else:
            raise ValueError('Unsupported distribution for likelihood')
        # Check for infinite code lengths
        if np.isinf(log_P_X1_given_E_base2):
            log_P_X1_given_E_base2 = 1e6  # e.g., LARGE_VALUE = 1e6
        # Compute P(X2 | X1)
        if dist_X2_given_X1 == 'bernoulli':
            p = theta_j_possible[int(X1)]
            P_X2_given_X1 = p if X2 == 1 else (1 - p)
            log_P_X2_given_X1_base2 = -np.log2(P_X2_given_X1)
        elif dist_X2_given_X1 == 'poisson':
            lambda_X2 = theta_j_possible[int(X1)]
            log_P_X2_given_X1 = adjusted_log_poisson_pmf(X2, lambda_X2, M)
            log_P_X2_given_X1_base2 = -log_P_X2_given_X1 / np.log(2)
        elif dist_X2_given_X1 == 'geometric':
            p = theta_j_possible[int(X1)]
            log_P_X2_given_X1 = adjusted_log_geometric_pmf(X2, p, M)
            log_P_X2_given_X1_base2 = -log_P_X2_given_X1 / np.log(2)
        elif dist_X2_given_X1 == 'gaussian':
            if sigma2_X2 is None:
                raise ValueError('sigma2_X2 must be provided for Gaussian distribution')
            sigma_X2 = np.sqrt(sigma2_X2)
            mu_X2_given_X1 = X1
            log_P_X2_given_X1 = adjusted_log_discretized_gaussian_pmf(X2, mu_X2_given_X1, sigma_X2, M)
            log_P_X2_given_X1_base2 = -log_P_X2_given_X1 / np.log(2)
        elif dist_X2_given_X1 == 'binomial':
            p = theta_j_possible[int(X1)]
            log_P_X2_given_X1 = log_binomial_pmf(X2, M+2, p)
            log_P_X2_given_X1_base2 = -log_P_X2_given_X1 / np.log(2)
        elif dist_X2_given_X1 == 'poisson_2':
            if X1 <= M/2:
                lambda_X2 = theta_j_possible[0] * (M+2)
            else:
                lambda_X2 = theta_j_possible[-1] * (M+2)
            log_P_X2_given_X1 = adjusted_log_poisson_pmf(X2, lambda_X2, M)
            log_P_X2_given_X1_base2 = -log_P_X2_given_X1 / np.log(2)
        elif dist_X2_given_X1 == 'geometric_2':
            if X1 <= M/2:
                p = theta_j_possible[0]
            else:
                p = theta_j_possible[-1]
            log_P_X2_given_X1 = adjusted_log_geometric_pmf(X2, p, M)
            log_P_X2_given_X1_base2 = -log_P_X2_given_X1 / np.log(2)
        else:
            raise ValueError('Unsupported distribution for likelihood')
        if np.isinf(log_P_X2_given_X1_base2):
            log_P_X2_given_X1_base2 = 1e6  # e.g., LARGE_VALUE = 1e6
        # Total negative log probability
        log_P_X += log_P_E_base2 + log_P_X1_given_E_base2 + log_P_X2_given_X1_base2

    return log_P_X

def find_best_partition_given_mechanisms(E_samples, X1_samples, X2_samples, N, k, M, mechanisms, theta_j_possible, dist_X1_given_E, dist_X2_given_X1, theta_E_hat, sigma2_X1_hat=None, sigma2_X2=None):
    partition = assign_environments_to_mechanisms(theta_E_hat, mechanisms)
    total_coding_length = compute_total_nll(
        partition, mechanisms, theta_j_possible, E_samples, X1_samples, X2_samples, N, k, M, dist_X1_given_E,
        dist_X2_given_X1, sigma2_X1=sigma2_X1_hat, sigma2_X2=sigma2_X2)
    return total_coding_length, partition

def save_total_nll(output_dir, distributions, n_samples_per_env, M, N, true_theta_E_list, ground_truth_k_list, k_values,
                               seed, theta_j_possible, method):
    np.random.seed(seed)
    results = {}
    dist_X1_given_E = distributions['X1']
    dist_X2_given_X1 = distributions['X2']

    for idx, true_theta_E in enumerate(true_theta_E_list):
        ground_truth_k = ground_truth_k_list[idx]
        true_theta_E = np.array(true_theta_E)

        print(f'\nProcessing n_samples_per_env={n_samples_per_env}, '
              f'Dist_X1_given_E={dist_X1_given_E}, Dist_X2_given_X1={dist_X2_given_X1}, '
              f'Ground Truth k={ground_truth_k}')
        print(f'Ground Truth theta_E: {true_theta_E}')
        sys.stdout.flush()

        total_nll_seeds_k = {k: [] for k in k_values}


        from data_generation import generate_data
        E_samples, X1_samples, X2_samples = generate_data(dist_X1_given_E, dist_X2_given_X1, M, N, n_samples_per_env, theta_j_possible, true_theta_E, seed=seed)
        theta_E_hat = estimate_theta_E_hat(E_samples, X1_samples, M, N, dist_X1_given_E)
        sigma2_X1_hat = 1
        sigma2_X2 = 1

        # Calculate total coding lengths
        for k in k_values:
            if method == 'greedy_search':
                best_total_coding_length = float('inf')
                max_possible_combinations = comb(len(theta_j_possible), k)
                all_possible_combinations = set(itertools.combinations(theta_j_possible, k))
                assert max_possible_combinations == len(all_possible_combinations)

                for mechanisms in all_possible_combinations:

                    total_coding_length, partition = find_best_partition_given_mechanisms(
                        E_samples, X1_samples, X2_samples, N, k, M, mechanisms, theta_j_possible, dist_X1_given_E,
                        dist_X2_given_X1, theta_E_hat, sigma2_X1_hat=sigma2_X1_hat, sigma2_X2=sigma2_X2)
                    if total_coding_length < best_total_coding_length:
                        best_total_coding_length = total_coding_length
                        best_partition = partition
                        best_mechanisms = mechanisms

                best_total_coding_length = compute_total_nll(
                        best_partition, best_mechanisms, theta_j_possible, E_samples, X1_samples,
                        X2_samples, N, k, M, dist_X1_given_E,
                    dist_X2_given_X1, sigma2_X1=sigma2_X1_hat, sigma2_X2=sigma2_X2)
                total_nll_seeds_k[k].append(best_total_coding_length)
            else:
                raise ValueError(f'Unsupported method: {method}')
            print(f"k={k}")
            print(f"best_partition={best_partition}")
            print(f"best_mechanisms={best_mechanisms}")
            sys.stdout.flush()
            results[f'theta_E_{idx}'] = total_nll_seeds_k

    # Save total_nll_seeds_k to file
    filename = f'{output_dir}/{dist_X1_given_E}_{dist_X2_given_X1}_samples{n_samples_per_env}_seed{seed}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved total coding lengths to {filename}')
    sys.stdout.flush()