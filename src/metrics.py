import numpy as np

def rmse_mae(X_true: np.ndarray, X_imp: np.ndarray, M: np.ndarray):
    diff = (X_imp - X_true)[M]
    rmse = float(np.sqrt((diff**2).mean()))
    mae  = float(np.abs(diff).mean())
    return rmse, mae