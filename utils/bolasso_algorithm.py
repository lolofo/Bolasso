import numpy as np
import sklearn
from sklearn.linear_model import Lasso, LinearRegression
from typing import Dict

def bolasso(X : np.ndarray, Y : np.ndarray, m : int, mu : float) -> Dict[str, np.ndarray]:
    """_summary_

    Args:
        X (np.ndarray): _description_
        Y (np.ndarray): _description_
        m (int): _description_
        mu (float): _description_
    """
    W = np.zeros(X.shape[1], m)

    for i in range(m):
        X_boot, Y_boot = sklearn.utils.resample(X, Y, random_state=0)
        model = Lasso(alpha=mu, fit_intercept=False)
        model.fit(X_boot, Y_boot)
        w_boot = model.coef_
        W[:,i] = w_boot

        
    W = np.sign(W)

    J = np.ones(X.shape[1], dtype=bool)
    current_W = W[: ,0]
    for i in range(1, m):
        next_W = W[:, i]
        buff = next_W == current_W
        res = res and buff
        current_W = next_W # update for the comparison


    final_model = LinearRegression()
    final_model.fit(X[:, J], Y[J])

    return {"coef" : final_model.coef_ , "J" : J}

    

     