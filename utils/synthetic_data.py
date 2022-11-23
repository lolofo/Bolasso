"""
The objective here is to generate synthetic data for
the bolasso algortihm

TODO : typing
"""

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from typing import Dict

def generate_synthetic_data(p : int, r : int, n : int) -> Dict[str, np.ndarray]:
    """ generate_synthetic_data

    The objective of this function is to generate synthetic data
    for the bolasso algorithm

    Args:
        p (int): dimension of the data
        r (int): sparse index
        n (int): number of sample data

    Returns :
        Dict(str, np.ndarray): contain the different data
    """
    assert p >= r, "error : the number of variables (p) must be greater than the sparse index (r)"

    # construction of the loading vectors
    loading_signs = np.random.choice(a=np.array([-1, 1]), size=r, replace=True)
    scaling = (1 - 1/3) * np.random.random_sample(r) + 1/3
    loading_vector = loading_signs * scaling

    w = np.zeros(p)
    w[:r] = loading_vector


    # construction of the variables
    X = np.zeros((n, p))
    Y = np.zeros(n)

    for i in range(n):
        G = np.random.randn(p,p)
        Q = G @ G.T
        d = np.sqrt(np.diag(1 / np.diagonal(Q)))
        S = d @ Q @ d.T # scale diagonal to unit >> force the variance of each component to one
        generator = multivariate_normal(np.zeros(p), S)
        x = generator.rvs()
        X[i, :] = x
        noise_generator = norm(loc=0, scale=0.1 * 2/3)
        Y[i] = x.T @ w + noise_generator.rvs()

    return {
        "X" : X,
        "Y" : Y,
        "w" : w
    }
    

if __name__ == "__main__" :
    print(help(generate_synthetic_data))

    data = generate_synthetic_data(16, 8, 40)

    print(data["X"].shape)
    print(data["Y"].shape)
    print(data["w"].shape)