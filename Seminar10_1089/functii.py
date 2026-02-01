import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def nan_replace_df(t: pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


def acp(x: np.ndarray, scal=True, ddof=0):
    n, m = x.shape
    x_ = x - np.mean(x, axis=0)
    if scal:
        x_ = x_ / np.std(x, axis=0, ddof=ddof)
    r_v = (1/(n-ddof))*x_.T@x_
    valp,vecp = np.linalg.eig(r_v)
    # print(valp)
    # print(vecp)
    k = np.flip(np.argsort(valp))
    # print(k)
    alpha = valp[k]
    a = vecp[:,k]
    return x_,r_v,alpha,a

def tabelare_varianta(alpha:np.ndarray):
    m = len(alpha)
    procent = alpha*100/sum(alpha)
    t = pd.DataFrame(data={
        "Varianta":alpha,
        "Varianta cumulata":np.cumsum(alpha),
        "Procent varianta":procent,
        "Procent cumulat":np.cumsum(procent)
    }, index=["C"+str(i+1) for i in range(m)])
    return t

def salvare_ndarray(x:np.ndarray,
                    nume_linii,
                    nume_coloane,
                    nume_index=None,
                    nume_fisier_output=None):
    temp = pd.DataFrame(x,nume_linii,nume_coloane)
    temp.index.name=nume_index
    if nume_fisier_output is not None:
        temp.to_csv(nume_fisier_output)
    return temp

def tabelare_varianta_factori(varianta):
    alpha = varianta[0]
    t = pd.DataFrame(
        data={
            "Varianta":alpha,
            "Varianta cumulata":np.cumsum(alpha),
            "Procent varianta":varianta[1]*100,
            "Procent cumulat":varianta[2]*100
        }, index=["F" + str(i+1) for i in range(len(alpha))]
    )
    return t