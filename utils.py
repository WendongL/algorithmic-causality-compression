import numpy as np
from functools import lru_cache
import math
from scipy.special import comb
import requests
import yaml

@lru_cache(maxsize=None)
def stirling_second_kind(n, k):
    if n == k == 0:
        return 1
    elif n == 0 or k == 0 or k > n:
        return 0
    else:
        return k * stirling_second_kind(n - 1, k) + stirling_second_kind(n - 1, k - 1)

@lru_cache(maxsize=None)
def compute_penalty(N, k, M):
    l_binom = np.log2(comb(M, k))
    l_k_factorial = np.log2(math.factorial(k))
    l_alpha = (l_binom + l_k_factorial + np.log2(stirling_second_kind(N, k)) + np.log2(k)) * 2 + 1
    return l_alpha

# Helper function to compute the KL divergence KL(P||Q)
@lru_cache(maxsize=None)
def kl_gaussian(P, Q):
    # P, Q need to be hashable. Tuple.
    P = np.array(P).reshape(2, 2)
    Q = np.array(Q).reshape(2, 2)
    # KL(P||Q) = 1/2[ log(|Σ_Q|/|Σ_P|) - 2 + Tr(Σ_Q^{-1} Σ_P) ]
    # Determinants
    det_P = P[0, 0] * P[1, 1] - P[0, 1] * P[1, 0]
    det_Q = Q[0, 0] * Q[1, 1] - Q[0, 1] * Q[1, 0]

    if det_P <= 0 or det_Q <= 0:
        # Numerical stability check, if covariance not valid:
        return np.inf

    # Inverse of Q
    inv_Q = np.array([[Q[1, 1], -Q[0, 1]],
                      [-Q[1, 0], Q[0, 0]]]) / det_Q

    # Trace(Σ_Q^{-1} Σ_P)
    prod = inv_Q @ P
    trace_term = prod[0, 0] + prod[1, 1]

    kl = 0.5 * (np.log(det_Q / det_P) - 2 + trace_term)
    return kl

def get_Sigma_matrices_for_kl(direction, sigma2_X1_est, sigma2_X2_est, linear_coeff_est, sigma2_X1_Q, sigma2_X2_Q, linear_coeff_Q):
    if direction == 'X1->X2':
        Sigma_P = tuple(np.array([
            [sigma2_X1_est, linear_coeff_est * sigma2_X1_est],
            [linear_coeff_est * sigma2_X1_est, linear_coeff_est ** 2 * sigma2_X1_est + sigma2_X2_est]
        ]).flatten())
        Sigma_Q = tuple(np.array([
            [sigma2_X1_Q, linear_coeff_Q * sigma2_X1_Q],
            [linear_coeff_Q * sigma2_X1_Q, linear_coeff_Q ** 2 * sigma2_X1_Q + sigma2_X2_Q]
        ]).flatten())
    elif direction == 'X2->X1':
        Sigma_P = tuple(np.array([
            [linear_coeff_est ** 2 * sigma2_X2_est + sigma2_X1_est, sigma2_X2_est * linear_coeff_est],
            [sigma2_X2_est * linear_coeff_est, sigma2_X2_est]
        ]).flatten())
        Sigma_Q = tuple(np.array([
            [linear_coeff_Q ** 2 * sigma2_X2_Q + sigma2_X1_Q, sigma2_X2_Q * linear_coeff_Q],
            [sigma2_X2_Q * linear_coeff_Q, sigma2_X2_Q]
        ]).flatten())
    else:
        raise ValueError('Unsupported causal direction')
    return Sigma_P, Sigma_Q

def send_ntfy_notification(topic, title, message):
    """
    Send a notification to the specified ntfy topic.
    """
    try:
        url = f"https://ntfy.sh/{topic}"
        data = {
            "title": title,
            "message": message,
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Notification sent successfully!")
        else:
            print(f"Failed to send notification: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending notification: {e}")

def load_config_from_yaml(yaml_file):
    """Load configuration from a YAML file."""
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config