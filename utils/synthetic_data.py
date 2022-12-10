import numpy as np
import os
import pickle
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
    G = np.random.randn(p, p)
    Q = G @ G.T
    d = np.sqrt(np.diag(1 / np.diagonal(Q)))
    Q = d @ Q @ d.T

    # proceed the criterion (2)
    Q1 = Q[r:, :][:, :r]
    Q2 = Q[:r, :][:, :r]

    if verbose :
        print(f"criterion (2) : {np.max(np.abs(Q1 @ np.linalg.inv(Q2) @ np.sign(w[:r])))}")



    for _ in tqdm(range(rep)):
        generator = multivariate_normal(np.zeros(p), Q)
        X = generator.rvs(n)
        noise_generator = norm(loc=0, scale=0.1 * np.sqrt(p * 13 / 27))
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


def load_data(p : int, r : int, n : int, rep : int, cache_path: str, criterion_2: bool = True, verbose : bool = True):
    """load data

    Args:
        p (int): _description_
        r (int): _description_
        n (int): _description_
        rep (int): _description_
        cache_path (str): _description_
        criterion_2 (bool, optional): _description_. Defaults to True.
        verbose (bool, optional): _description_. Defaults to True.
    """

    if criterion_2:
        cache_path = os.path.join(cache_path, "criterion_2_ok.pkl")
    else :
        cache_path = os.path.join(cache_path, "criterion_2_non_ok.pkl")

    if os.path.exists(cache_path):
        if verbose:
            print(f">> data loaded at : \n >> {cache_path}")
        with open(cache_path, 'rb') as f:
            res = pickle.load(f)

    else :
        if verbose:
            print(f">> create data with criterion (2) to : {criterion_2}")
        cont = True
        while cont :

            res = generate_synthetic_data(p, r, n , rep, verbose=False)
            tmp = res["crit_2"]
            
            tmp = tmp <= 1 # do we have a criterion 2 respect
            print("\t >> ", tmp)

            if criterion_2 and tmp :
                cont = False
            if (not criterion_2) and (not tmp):
                cont = False
        if verbose:
            print(f">> data created with criterion (2) : {res['crit_2']}")

        with open(cache_path, 'wb') as f:
            pickle.dump(res, f)

        if verbose:
            print(">> data saved : ready to use and re-use")

    return res


    
        
