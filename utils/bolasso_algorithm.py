import numpy as np
import sklearn
from sklearn.linear_model import Lasso, LinearRegression
from typing import Dict

def bolasso(X : np.ndarray, Y : np.ndarray, m : int, mu : float, verbose : bool = True) -> Dict[str, np.ndarray]:
    
    """ bolasso

    Args:
        X (np.ndarray): The dataset we'll use
        Y (np.ndarray): the target variable
        m (int): number of bootstrap
        mu (float): regularization coefficient for the lasso regularization
    """
    
    
    W = np.zeros((X.shape[1], m))
    verboseprint = print if verbose else lambda *a, **k: None
    for i in range(m):
        X_boot, Y_boot = sklearn.utils.resample(X, Y)
        model = Lasso(alpha=mu, fit_intercept=False)
        model.fit(X_boot, Y_boot)
        w_boot = model.coef_
        W[:,i] = w_boot

        
    W = np.sign(W)

    J = np.all(np.abs(W) > 0, axis=1)
    
    if all(~J):
        verboseprint("J is empty array")
        coef = np.zeros(X.shape[1])
    else:
        final_model = LinearRegression()
        final_model.fit(X[:, J], Y)

        coef = np.zeros(X.shape[1])
        coef[J] = final_model.coef_ 

    return {"J" : J, "coef" : coef}
