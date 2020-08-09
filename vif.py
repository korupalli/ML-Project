#VIF
from scipy.linalg.lapack import spotrf, sposv
import numpy as np
import pandas as pd

def variance_inflation_factor(X):
    """
    Calculates the variance inflation factor for each column in a dataframe.
    Input:
        X: Dataframe
    Output:
        Dataframe of variance inflation values for each column.
    """
    isDataframe = type(X) is pd.DataFrame
    if isDataframe:
        columns = X.columns
        X = X.values
    n, p = X.shape

    swap = np.arange(p)
    np.random.shuffle(swap)

    XTX = X.T @ X
    XTX = XTX[swap][:,swap]

    select = np.ones(p, dtype = bool)

    temp = XTX.copy().T
    error = 1
    largest = XTX.diagonal().max() // 2
    add = largest
    maximum = np.finfo(np.float32).max

    while error != 0:
        C, error = spotrf(a = temp)
        if error != 0:
            error -= 1
            select[error] = False
            temp[error, error] += add
            error += 1

            add += np.random.randint(1,30)
            add *= np.random.randint(30,50)
        if add > maximum:
            add = largest

    VIF = np.empty(p, dtype = np.float32)
    means = np.mean(X, axis = 0)[swap]

    for i in range(p):
        curr = select.copy()
        s = swap[i]

        if curr[i] == False:
            VIF[s] = np.inf
            continue
        curr[i] = False

        XX = XTX[curr]
        xtx = XX[:, curr]
        xty = XX[:,i]
        y_x = X[:,s]

        theta_x = sposv(xtx, xty)[1]
        y_hat = X[:,swap[curr]] @ theta_x

        SS_res = y_x-y_hat
        SS_res = np.einsum('i,i', SS_res, SS_res)
        #SS_res = np.sum((y_x - y_hat)**2)

        SS_tot = y_x - means[i]
        SS_tot = np.einsum('i,i', SS_tot, SS_tot)
        #SS_tot = np.sum((y_x - np.mean(y_x))**2)
        if SS_tot == 0:
            R2 = 1
            VIF[s] = np.inf
        else:
            R2 = 1 - (SS_res/SS_tot)
            VIF[s] = 1/(1-R2)
        del XX, xtx, xty, y_x, theta_x, y_hat
    if isDataframe:
        df_vif = pd.DataFrame({"vif": VIF})
        df_vif = df_vif.set_index(columns)
        return df_vif
    return VIF