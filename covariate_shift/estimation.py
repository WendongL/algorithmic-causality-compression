import numpy as np

def estimate_theta_E_hat(E_samples, X1_samples, M, N, dist_X1_given_E):
    theta_E_hat = np.zeros(N)
    for E in range(N):
        idx_E = (E_samples == E)
        X1_E_samples = X1_samples[idx_E]
        if len(X1_E_samples) > 0:
            if dist_X1_given_E == 'bernoulli':
                theta_E_hat[E] = np.mean(X1_E_samples)
            elif dist_X1_given_E == 'poisson':
                lambda_E_hat = np.mean(X1_E_samples)
                theta_E_hat[E] = lambda_E_hat / (M+2)
            elif dist_X1_given_E == 'geometric':
                mean_X1 = np.mean(X1_E_samples)
                theta_E_hat[E] = 1 / (mean_X1 + 1e-10)
                theta_E_hat[E] = np.clip(theta_E_hat[E], 0.01, 0.99)
            elif dist_X1_given_E == 'gaussian':
                mu_E_hat = np.mean(X1_E_samples)
                theta_E_hat[E] = mu_E_hat / (M+2)
            elif dist_X1_given_E == 'binomial':
                theta_E_hat[E] = np.mean(X1_E_samples) / (M+2) # Suppose (M+2) is the number of trials
            else:
                raise ValueError('Unsupported distribution for estimation')
        else:
            theta_E_hat[E] = 0.0

    return theta_E_hat
