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

def standardizare_zscore(x: np.ndarray):
    # Standardizare pe coloane: (x - media) / deviatia standard
    # Evitam impartirea la zero daca o coloana are varianta 0
    medii = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0, ddof=0)
    std[std == 0] = 1.0
    return (x - medii) / std

def calcul_partitie(h: np.ndarray, k=None):
    m = h.shape[0]
    n = m + 1
    if k is None:
        # Partitia optimala. Se aplica Elbow pe distantele de legare
        diferente = h[1:, 2] - h[:m - 1, 2]
        j = np.argmax(diferente) + 1
        k = n - j
    else:
        # Partitie cu k clusteri
        j = n - k

    color_threshold = (h[j, 2] + h[j - 1, 2]) / 2
    c = np.arange(n)  # partitie singleton

    for i in range(j):
        k1 = h[i, 0]
        k2 = h[i, 1]
        c[c == k1] = n + i
        c[c == k2] = n + i

    partitie = ["C" + str(i + 1) for i in pd.Categorical(c).codes]
    return k, color_threshold, np.array(partitie)


def distante_euclidiene(x: np.ndarray):
    # matricea distantelor euclidiene intre toate perechile de observatii
    n = x.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dij = np.linalg.norm(x[i] - x[j])
            d[i, j] = dij
            d[j, i] = dij
    return d

def silhouette_scor_si_valori(d: np.ndarray, partitii: np.ndarray):
    # d: matrice distante (n x n)
    # partitii: vector etichete cluster (ex: ['C1','C2',...]) lungime n
    n = len(partitii)
    et = np.array(partitii)
    clustere = np.unique(et)

    s = np.zeros(n)

    for i in range(n):
        c_i = et[i]
        # a(i): distanta medie fata de punctele din acelasi cluster (fara i)
        same = np.where(et == c_i)[0]
        same = same[same != i]
        if same.size == 0:
            a = 0.0
        else:
            a = d[i, same].mean()

        # b(i): minima dintre distantele medii fata de celelalte clustere
        b = np.inf
        for c in clustere:
            if c == c_i:
                continue
            other = np.where(et == c)[0]
            if other.size == 0:
                continue
            b_c = d[i, other].mean()
            if b_c < b:
                b = b_c

        # s(i)
        if max(a, b) == 0 or np.isinf(b):
            s[i] = 0.0
        else:
            s[i] = (b - a) / max(a, b)

    return s.mean(), s


def axe_principale(x: np.ndarray):
    # x: matrice standardizata (n x p)
    # returneaza: scoruri (n x p), valori proprii, vectori proprii
    # calcul matrice de covarianta
    c = np.cov(x, rowvar=False)
    # descompunere spectrala
    valori, vectori = np.linalg.eigh(c)
    # ordonare descrescator dupa valori proprii
    idx = np.argsort(valori)[::-1]
    valori = valori[idx]
    vectori = vectori[:, idx]
    # scoruri (proiectii)
    scoruri = x @ vectori
    return scoruri, valori, vectori

