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


def salvare_ndarray(x: np.ndarray, nume_linii, nume_coloane, nume_fisier_output="out.csv"):
    temp = pd.DataFrame(x, nume_linii, nume_coloane)
    temp.to_csv(nume_fisier_output)


def calcul_partitie(h: np.ndarray, k=None):
    m = h.shape[0]
    n = m + 1
    if k is None:
        # Partitia optimala. Se aplica Elbow
        diferente = h[1:, 2] - h[:m - 1, 2]
        j = np.argmax(diferente) + 1
        k = n - j
    else:
        # Calcul partitie cu k clusteri
        j = n - k
    color_threshold = (h[j, 2] + h[j - 1, 2]) / 2
    c = np.arange(n)  # Partitia singleton
    for i in range(j):
        k1 = h[i, 0]
        k2 = h[i, 1]
        c[c == k1] = n + i
        c[c == k2] = n + i
    partitie = ["C"+str(i+1) for i in pd.Categorical(c).codes]
    return k, color_threshold, np.array(partitie)
