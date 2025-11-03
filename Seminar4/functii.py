import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)

def acp(x:np.ndarray, scal = True, ddof=0):
    n, m = x.shape
    x_ = x - np.mean(x, axis = 0)
    if scal:
        x_ = x_ / np.std(x, axis = 0, ddof=ddof)
        #x_.T = transpusa lui x_
        # @ = operator de înmulțire matriceală (echivalent cu np.matmul(x_.T, x_))
        # (1 / (n - ddof)) = factor scalar de normalizare
    r_v = (1/(n - ddof)) * x_.T @ x_
    valp, vecp = np.linalg.eig(r_v) # valori proprii, vectori proprii
    # print(valp)
    # print(vecp)
    k = np.flip(np.argsort(valp))
    print(k)
    alpha = valp[k]
    a = vecp[:,k]
    return alpha, a

def tabelare_varianta(alpha:np.ndarray):
    n = len(alpha)
    procent = alpha*100/sum(alpha)
    t = pd.DataFrame(data = {
        "Varianta": alpha,
        "Varianta cumulata": np.cumsum(alpha),
        "Procent varianta": alpha*100/sum(alpha),
        "Procent cumulat": np.cumsum(procent)
    }, index=["C" + str(i) for i in range(n)])
    return t