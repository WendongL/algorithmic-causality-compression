import numpy as np

def generate_data(distributions, N, n_samples_per_env, true_theta_E, seed=None):
    if seed is not None:
        np.random.seed(seed)

    X_samples_all_env = []
    if distributions == 'c_gaussian':
        for E in range(N):
            sigma2_X1, sigma2_X2, linear_coeff = true_theta_E[E]
            Sigma = np.array([
                [sigma2_X1, sigma2_X1 * linear_coeff],
                [sigma2_X1 * linear_coeff, linear_coeff**2 * sigma2_X1 + sigma2_X2]
            ])
            X_samples = np.random.multivariate_normal(mean=[0, 0], cov=Sigma, size=n_samples_per_env)
            X_samples_all_env.append(X_samples)
    else:
        raise NotImplementedError("Only 'c_gaussian' distributions are currently supported.")

    return np.array(X_samples_all_env)  # shape (N, n_samples_per_env, 2)
