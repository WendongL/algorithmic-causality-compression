import numpy as np
import sys
import os
current_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current_dir, '..'))
from utils import kl_gaussian, get_Sigma_matrices_for_kl

def assign_environments_to_mechanisms(theta_E_hat, mechanisms, direction):
    sigma2_X1_hat_E = theta_E_hat['sigma2_X1_hat_E']
    sigma2_X2_hat_E = theta_E_hat['sigma2_X2_hat_E']
    linear_coeff_hat_E = theta_E_hat['linear_coeff_hat_E']

    N = len(sigma2_X1_hat_E)

    k_sigma2_X1 = len(mechanisms[0])
    k_sigma2_X2 = len(mechanisms[1])
    k_linear_coeff = len(mechanisms[2])

    partition_sigma2_X1 = [[] for _ in range(k_sigma2_X1)]
    partition_sigma2_X2 = [[] for _ in range(k_sigma2_X2)]
    partition_linear_coeff = [[] for _ in range(k_linear_coeff)]

    envs = range(N)
    sigma2_X1_candidates_list = mechanisms[0]
    sigma2_X2_candidates_list = mechanisms[1]
    linear_coeff_candidates_list = mechanisms[2]

    kl_value_all_env = 0
    for E in envs:
        sigma2_X1_P = sigma2_X1_hat_E[E]
        sigma2_X2_P = sigma2_X2_hat_E[E]
        linear_coeff_P = linear_coeff_hat_E[E]
        if direction == 'X1->X2':
            Sigma_P = tuple(np.array([
                [sigma2_X1_P, linear_coeff_P * sigma2_X1_P],
                [linear_coeff_P * sigma2_X1_P, linear_coeff_P ** 2 * sigma2_X1_P + sigma2_X2_P]
            ]).flatten())
        elif direction == 'X2->X1':
            Sigma_P = tuple(np.array([
                [linear_coeff_P ** 2 * sigma2_X2_P + sigma2_X1_P, sigma2_X2_P * linear_coeff_P],
                [sigma2_X2_P * linear_coeff_P, sigma2_X2_P]
            ]).flatten())
        else:
            raise ValueError('Unsupported causal direction')

        kl_values = np.zeros((len(sigma2_X1_candidates_list), len(sigma2_X2_candidates_list), len(linear_coeff_candidates_list)))
        for idx1, sigma2_X1_candidate in enumerate(sigma2_X1_candidates_list):
            for idx2, sigma2_X2_candidate in enumerate(sigma2_X2_candidates_list):
                for idx3, linear_coeff_candidate in enumerate(linear_coeff_candidates_list):
                    if direction == 'X1->X2':
                        Sigma_Qi = tuple(np.array([
                            [sigma2_X1_candidate, linear_coeff_candidate * sigma2_X1_candidate],
                            [linear_coeff_candidate * sigma2_X1_candidate, linear_coeff_candidate ** 2 * sigma2_X1_candidate + sigma2_X2_candidate]
                        ]).flatten())
                    elif direction == 'X2->X1':
                        Sigma_Qi = tuple(np.array([
                            [linear_coeff_candidate ** 2 * sigma2_X2_candidate + sigma2_X1_candidate, sigma2_X2_candidate * linear_coeff_candidate],
                            [sigma2_X2_candidate * linear_coeff_candidate, sigma2_X2_candidate]
                        ]).flatten())
                    else:
                        raise ValueError('Unsupported causal direction')
                    kl_val = kl_gaussian(Sigma_P, Sigma_Qi)
                    kl_values[idx1, idx2, idx3] = kl_val
        idx_min = np.unravel_index(np.argmin(kl_values), kl_values.shape)
        kl_value_all_env += kl_values[idx_min[0], idx_min[1], idx_min[2]]
        partition_sigma2_X1[idx_min[0]].append(E)
        partition_sigma2_X2[idx_min[1]].append(E)
        partition_linear_coeff[idx_min[2]].append(E)

    return partition_sigma2_X1, partition_sigma2_X2, partition_linear_coeff, kl_value_all_env


def direct_assign_environments_to_mechanisms(theta_E_hat, zeta2_X1_possible, zeta2_X2_possible, coeff_possible, direction):
    """
    Directly assigns environments to mechanisms by finding the closest match
    for each environment's estimated parameter to the list of possible mechanisms.
    The function also computes and accumulates the Kullback-Leibler divergence
    between the estimated and assigned mechanisms for all environments.

    Args:
        theta_E_hat (dict): Dictionary containing estimated parameters for
            each environment including 'sigma2_X1_hat_E', 'sigma2_X2_hat_E',
            and 'linear_coeff_hat_E'.
        zeta2_X1_possible (list): List of possible values for sigma2_X1 mechanisms.
        zeta2_X2_possible (list): List of possible values for sigma2_X2 mechanisms.
        coeff_possible (list): List of possible values for linear coefficients.
        direction (str): Causal direction, either 'X1->X2' or 'X2->X1'.

    Returns:
        tuple: A tuple containing three lists representing the partition of
        environments for each parameter (sigma2_X1, sigma2_X2, linear_coeff)
        and the total accumulated KL divergence across all environments.
    """
    sigma2_X1_hat_E = theta_E_hat['sigma2_X1_hat_E']
    sigma2_X2_hat_E = theta_E_hat['sigma2_X2_hat_E']
    linear_coeff_hat_E = theta_E_hat['linear_coeff_hat_E']

    N = len(sigma2_X1_hat_E)

    partition_sigma2_X1 = [[] for _ in range(len(zeta2_X1_possible))]
    partition_sigma2_X2 = [[] for _ in range(len(zeta2_X2_possible))]
    partition_linear_coeff = [[] for _ in range(len(coeff_possible))]

    kl_value_all_env = 0

    for E in range(N):
        # Find closest sigma2_X1
        idx_closest_X1 = np.argmin(np.abs(np.array(zeta2_X1_possible) - sigma2_X1_hat_E[E]))
        # Find closest sigma2_X2
        idx_closest_X2 = np.argmin(np.abs(np.array(zeta2_X2_possible) - sigma2_X2_hat_E[E]))
        # Find closest linear_coeff
        idx_closest_coeff = np.argmin(np.abs(np.array(coeff_possible) - linear_coeff_hat_E[E]))

        partition_sigma2_X1[idx_closest_X1].append(E)
        partition_sigma2_X2[idx_closest_X2].append(E)
        partition_linear_coeff[idx_closest_coeff].append(E)

        # Compute KL
        Sigma_P, Sigma_Q = get_Sigma_matrices_for_kl(
            direction,
            sigma2_X1_hat_E[E], sigma2_X2_hat_E[E], linear_coeff_hat_E[E],
            zeta2_X1_possible[idx_closest_X1], zeta2_X2_possible[idx_closest_X2], coeff_possible[idx_closest_coeff]
        )
        kl_val = kl_gaussian(Sigma_P, Sigma_Q)
        kl_value_all_env += kl_val

    return partition_sigma2_X1, partition_sigma2_X2, partition_linear_coeff, kl_value_all_env