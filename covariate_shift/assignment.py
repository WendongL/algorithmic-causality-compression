import numpy as np

def assign_environments_to_mechanisms(theta_E_hat, mechanisms):
    N = len(theta_E_hat)
    k = len(mechanisms)
    partition = [[] for _ in range(k)]
    envs = set(range(N))
    for E in envs:
        distances = np.abs(mechanisms - theta_E_hat[E])
        idx_min = np.argmin(distances)
        partition[idx_min].append(E)
    return partition