import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TipSubiectPCA.solutie import eigvals


# Functii utilitare
def zscore(X):
    """
    Standardizare Z-score:
    fiecare variabila -> media 0, deviatia standard 1
    Necesara pentru ACP (PCA)
    """
    return (X - X.mean()) / X.std(ddof=0)

# Citire date
df = pd.read_csv('./dataIN/GlobalIndicators_Proc_2021.csv')

# Country = identificator (nu intra in calcule)
df = df.set_index("Country")

# A) Export - Import (Excedent comercial)

# Excedent = Exporturi - Importuri
df["TradeBalance"] = df["Exports"] - df["Imports"]

# Pastram doar tarile cu excedent pozitiv
trade = df[df["TradeBalance"] > 0][["TradeBalance"]]

# Sortare descrescatoare
trade = trade.sort_values(by="TradeBalance", ascending=False)

trade.to_csv('./dataOUT/output.csv')

# B) Matricea de corelatii
X = df.select_dtypes(include=[np.number])

corr = X.corr()
corr.to_csv('./dataOUT/corr.csv')

# C) Analiza in componente principale (ACP)

# --- Standardizare
Xstd = zscore(X)

# --- Matrice de covarianta
cov = np.cov(Xstd.T)

# --- Valori si vectori proprii
eigvals, eigvecs = np.linalg.eig(cov)

# Sortare descrescatoare
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

# --- Scree plot
plt.figure()
plt.plot(np.arange(1, len(eigvals)+1), eigvals, marker="o")
plt.axhline(y=1, linestyle="--")
plt.xlabel("Componenta principala")
plt.ylabel("Varianta explicata")
plt.title("Scree plot - Criteriul Kaiser")
plt.show()

# Corelatii variabile - componente

# corelatii = vectori proprii * sqrt(valoare proprie)
corr_pc = eigvecs * np.sqrt(eigvals)

corr_pc_df = pd.DataFrame(
    corr_pc,
    index=X.columns,
    columns=[f"PC{i+1}" for i in range(len(eigvals))]
)

corr_pc_df.to_csv('./dataOUT/corr_pc.csv')

# Cosinusuri patrate (cos^2)

cos2 = corr_pc_df ** 2
cos2.to_csv('./dataOUT/cos2.csv')

# Analiza discriminanta

"""
1. Avem 3 variabile X1, X2, X3
2. Avem 4 clase A, B, C, D
3. Pentru fiecare clasa:
    - calculam vectorul mediu
4. Calculam matricea de covarianta totala
5. Pentru o instanta noua x:
    - calculam distanta Mahalanobis fata de fiecare centru de clasa
    - atribuim clasa cu distanta minima
    
Formula distantei: 
d(x, m) = (x - m)' * Cov^-1 * (x - m)



"""
