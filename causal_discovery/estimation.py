import numpy as np

def estimate_theta_E_hat(X_samples_all_env, N, distributions, causal_direction):
    if distributions == 'c_gaussian':
        sigma2_X1_hat_E = np.zeros(N)
        sigma2_X2_hat_E = np.zeros(N)
        linear_coeff_hat_E = np.zeros(N)
        theta_E_hat = {}
        for E in range(N):
            cov_E_hat = np.cov(X_samples_all_env[E], rowvar=False)
            if causal_direction == 'X1->X2':
                sigma2_X1_hat = cov_E_hat[0,0]
                linear_coeff_hat = cov_E_hat[0,1] / sigma2_X1_hat
                sigma2_X2_hat = cov_E_hat[1,1] - linear_coeff_hat**2 * sigma2_X1_hat
            elif causal_direction == 'X2->X1':
                sigma2_X2_hat = cov_E_hat[1,1]
                linear_coeff_hat = cov_E_hat[0,1] / sigma2_X2_hat
                sigma2_X1_hat = cov_E_hat[0,0] - linear_coeff_hat**2 * sigma2_X2_hat
            else:
                raise ValueError('Unsupported causal direction')

            sigma2_X1_hat_E[E] = sigma2_X1_hat
            sigma2_X2_hat_E[E] = sigma2_X2_hat
            linear_coeff_hat_E[E] = linear_coeff_hat

        theta_E_hat['sigma2_X1_hat_E'] = sigma2_X1_hat_E
        theta_E_hat['sigma2_X2_hat_E'] = sigma2_X2_hat_E
        theta_E_hat['linear_coeff_hat_E'] = linear_coeff_hat_E
    else:
        raise ValueError('Unsupported distribution for estimation')
    return theta_E_hat
