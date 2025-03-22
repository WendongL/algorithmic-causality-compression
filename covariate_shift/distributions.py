import numpy as np
from scipy.stats import poisson, geom, norm
import math


def adjusted_log_poisson_pmf(x, lambda_E, M):
    """
    Compute the adjusted log Poisson PMF.

    For x < M-2, the log PMF is computed directly.  For x >= M-2, the
    log(1 - CDF) is computed, since the PMF is not defined for x > M-2.

    Parameters
    ----------
    x : int
        The value of the random variable.
    lambda_E : float
        The rate parameter of the Poisson distribution.
    M : int
        The maximum possible value of x.

    Returns
    -------
    log_P : float
        The adjusted log Poisson PMF.
    """
    if x < M-1:
        # Use log PMF
        log_P = -lambda_E + x * np.log(lambda_E) - math.lgamma(x + 1)
    elif x == M-1:
        # For x >= M-1, compute log(1 - CDF)
        cdf = poisson.cdf(M-2, lambda_E)
        if cdf >= 1.0:
            log_P = -np.inf
        else:
            log_P = np.log(1 - cdf)
    else:
        log_P = -np.inf
    return log_P

def adjusted_log_geometric_pmf(x, p, M):
    if x < M-1:
        # Geometric PMF: P(X=x) = (1-p)^x * p
        log_P = x * np.log(1 - p) + np.log(p)
    elif x == M-1:
        cdf = geom.cdf(M-1, p)
        if cdf >= 1.0:
            log_P = -np.inf
        else:
            log_P = np.log(1 - cdf)
    else:
        log_P = -np.inf
    return log_P

def adjusted_log_discretized_gaussian_pmf(x, mu, sigma, M):

    lower = x - 0.5
    upper = x + 0.5
    if x == 0:
        lower = -np.inf
    if x == M-1:
        upper = np.inf
    cdf_lower = norm.cdf(lower, loc=mu, scale=sigma)
    cdf_upper = norm.cdf(upper, loc=mu, scale=sigma)
    P = cdf_upper - cdf_lower
    if P <= 0:
        log_P = -np.inf
    else:
        log_P = np.log(P)
    return log_P

def log_binomial_pmf(x, n, p):
    """
    Computes the log probability of the binomial PMF.

    Parameters
    ----------
    x : int
        Number of successes
    n : int
        Number of trials
    p : float
        Probability of success

    Returns
    -------
    log_P : float
        Log probability of the binomial PMF
    """

    log_P = x * np.log(p) + (n - x) * np.log(1 - p) + math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
    return log_P