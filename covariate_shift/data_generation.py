import numpy as np

def generate_data(dist_X1_given_E, dist_X2_given_X1, M, N, n_samples_per_env, theta_j_possible, true_theta_E, seed=None):
    if seed is not None:
        np.random.seed(seed)

    E_samples = np.repeat(np.arange(N), n_samples_per_env)
    np.random.shuffle(E_samples)

    # Define standard deviations
    sigma_X1 = 1.0
    sigma_X2 = 1.0
    sigma2_X2 = sigma_X2 ** 2

    # Generate X1_samples
    X1_samples = []
    for E in E_samples:
        theta_E_val = true_theta_E[E]
        if dist_X1_given_E == 'bernoulli':
            X1 = np.random.binomial(n=1, p=theta_E_val)
        elif dist_X1_given_E == 'poisson':
            lambda_E = theta_E_val * (M + 2)
            X1 = np.random.poisson(lam=lambda_E)
            X1 = min(X1, M - 1)  # Since the range of X1 is [0, M-1], in total M possible values
        elif dist_X1_given_E == 'geometric':
            X1 = np.random.geometric(p=theta_E_val) - 1
            X1 = min(X1, M - 1)
        elif dist_X1_given_E == 'gaussian':
            X1 = np.random.normal(loc=theta_E_val * (M + 2), scale=sigma_X1)
            X1 = int(np.round(X1))
            X1 = min(X1, M - 1)
        else:
            raise ValueError('Unsupported distribution for X1 given E')
        X1_samples.append(X1)
    X1_samples = np.array(X1_samples)

    # Generate X2_samples
    X2_samples = []
    for X1 in X1_samples:
        if dist_X2_given_X1 == 'bernoulli':
            p = theta_j_possible[int(X1)]
            X2 = np.random.binomial(n=1, p=p)
        elif dist_X2_given_X1 == 'poisson':
            lambda_X2 = theta_j_possible[int(X1)] * (M + 2)
            X2 = np.random.poisson(lam=lambda_X2)
            X2 = min(X2, M - 1)
        elif dist_X2_given_X1 == 'geometric':
            p = theta_j_possible[int(X1)]
            X2 = np.random.geometric(p=p) - 1
            X2 = min(X2, M - 1)
        elif dist_X2_given_X1 == 'gaussian':
            X2 = np.random.normal(loc=X1, scale=sigma_X2)
            X2 = int(np.round(X2))
            X2 = min(max(X2, 0), M - 1)
        elif dist_X2_given_X1 == 'poisson_2':
            if X1 <= M / 2:
                lambda_X2 = theta_j_possible[0] * (M + 2)
            else:
                lambda_X2 = theta_j_possible[-1] * (M + 2)
            X2 = np.random.poisson(lam=lambda_X2)
            X2 = min(X2, M - 1)
        elif dist_X2_given_X1 == 'geometric_2':
            if X1 <= M / 2:
                p = theta_j_possible[0]
            else:
                p = theta_j_possible[-1]
            X2 = np.random.geometric(p=p) - 1
            X2 = min(X2, M - 1)
        else:
            raise ValueError('Unsupported distribution for X2 given X1')
        X2_samples.append(X2)
    X2_samples = np.array(X2_samples)

    return E_samples, X1_samples, X2_samples