import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from tqdm import tqdm
from typing import Dict

def generate_synthetic_data(p : int, r : int, n : int, rep : int, verbose : bool = True) -> Dict[str, np.ndarray]:
    """ generate_synthetic_data

    The objective of this function is to generate synthetic data
    for the bolasso algorithm

    Args:
        p (int): dimension of the data
        r (int): sparse index
        n (int): number of sample data
        rep (int) : number of time we replicate the data

    Returns :
        Dict(str, np.ndarray): contain the different data
    """
    assert p >= r, "error : the number of variables (p) must be greater than the sparse index (r)"

    X_containers = []
    Y_containers = []
    W_containers = []
    # construction of the loading vectors
    loading_signs = np.random.choice(a=np.array([-1, 1]), size=r, replace=True)
    scaling = (1 - 1/3) * np.random.random_sample(r) + 1/3
    loading_vector = loading_signs * scaling

    w = np.zeros(p)
    w[:r] = loading_vector


    # construction of the variables
    X = np.zeros((n, p))
    Y = np.zeros(n)

    # generation of the covariance matrix
    G = np.random.randn(p,p)
    Q = G @ G.T
    d = np.sqrt(np.diag(1 / np.diagonal(Q)))
    S = d @ Q @ d.T # scale diagonal to unit >> force the variance of each component to one

    # proceed the criterion (2)
    Q1 = Q[r:, :][:, :r]
    Q2 = Q[:r, :][:, :r]

    if verbose :
        print(f"criterion (2) : {np.max(np.abs(Q1 @ np.linalg.inv(Q2) @ np.sign(w[:r])))}")



    for _ in tqdm(range(rep)):
        generator = multivariate_normal(np.zeros(p), S)
        X = generator.rvs(n)
        noise_generator = norm(loc=0, scale=0.1 * 2/3)
        Y = X @ w + noise_generator.rvs(n)

        X_containers.append(X)
        Y_containers.append(Y)
        W_containers.append(w)

    return {
        "X" : X_containers,
        "Y" : Y_containers,
        "w" : W_containers,
        "crit_2" : np.max(np.abs(Q1 @ np.linalg.inv(Q2) @ np.sign(w[:r]))),
    }
