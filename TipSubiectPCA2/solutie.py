import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 0) SETĂRI GENERALE (foldere IN / OUT)
# ============================================================
INPUT_DIR = "dataIN"
OUTPUT_DIR = "dataOUT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1) FUNCȚII AJUTĂTOARE (explicate clar)
# ============================================================

def per_capita(value_series: pd.Series, population_series: pd.Series) -> pd.Series:
    """
    Calculează "pe locuitor" (per inhabitant / per capita).

    Formula:
        valoare_pe_locuitor = valoare_totala / populatie

    De ce facem asta?
    - Ca să comparăm localități de mărimi diferite.
    - Altfel, o localitate mare va avea mereu valori totale mai mari doar fiindcă are mai mulți oameni.

    Protecții:
    - Dacă populația e 0 sau lipsă, împărțirea nu are sens => punem NaN.
    """
    pop = population_series.replace(0, np.nan)
    return value_series / pop


def percent_structure(df_sum_expenses: pd.DataFrame) -> pd.DataFrame:
    """
    Transformă cheltuielile totale (pe județ) în procente din total.

    Input:
      df_sum_expenses: tabel pe județ cu sume pe categorii C1..C8 (ex: după groupby.sum)

    Output:
      un tabel pe județ cu procentele fiecărei categorii în total cheltuieli.

    Formula:
        pondere_Ci(%) = Ci / (C1+...+C8) * 100

    De ce e cerut asta?
    - Profesorul vrea "structura" cheltuielilor, adică din total,
      cât % se duce pe personal, educație, sănătate etc.
    """
    total = df_sum_expenses.sum(axis=1).replace(0, np.nan)  # total cheltuieli pe județ
    return df_sum_expenses.div(total, axis=0) * 100


def zscore_standardize(X: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizare Z-score pe fiecare coloană (variabilă):

        z = (x - media) / deviatia_standard

    De ce standardizăm înainte de PCA?
    - PCA este bazată pe varianțe/covarianțe.
    - Dacă variabilele au scări diferite (ore, %, număr), cele "mari" domină analiza.
    - Standardizarea pune toate variabilele pe aceeași scară (media 0, std 1).

    Observație:
    - ddof=0 (deviație standard de populație) e folosit des în seminarii.
    """
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0).replace(0, 1)  # protecție: dacă std=0, evităm împărțirea la 0
    return (X - means) / stds


def covariance_matrix(Xstd: pd.DataFrame) -> pd.DataFrame:
    """
    Calculează matricea de varianță-covarianță pentru variabilele STANDARDIZATE.

    În numpy:
      np.cov(Xstd.T)  -> fiindcă vrem covarianța între coloane (variabile),
                         nu între linii (observații).

    Rezultatul este o matrice pătrată p x p (p = număr variabile),
    unde:
      - pe diagonală ai varianțele (≈ 1 pentru date standardizate)
      - în afara diagonalei ai covarianțe (măsoară cum variază împreună două variabile)
    """
    cov = np.cov(Xstd.values.T, ddof=0)
    return pd.DataFrame(cov, index=Xstd.columns, columns=Xstd.columns)


def pca_scores_from_cov(Xstd: pd.DataFrame, Cov: pd.DataFrame):
    """
    Face PCA "manual" (stil seminar):
    1) Decompoziție spectrală pe matricea de covarianță:
         Cov * v = lambda * v
       - lambda = valori proprii (varianța explicată de componentă)
       - v = vectori proprii (direcțiile/componentelor)

    2) Sortăm valorile proprii descrescător (PC1 explică cel mai mult)

    3) Calculăm scorurile (coordonatele observațiilor în spațiul PC):
         Scores = Xstd * eigvecs

    Return:
      eigvals (vector), eigvecs (matrice), Scores (DataFrame)
    """
    eigvals, eigvecs = np.linalg.eig(Cov.values)

    # Sortare descrescătoare după eigvals
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Scoruri PCA (observații proiectate pe componente)
    scores = Xstd.values @ eigvecs
    scores_df = pd.DataFrame(
        scores,
        index=Xstd.index,
        columns=[f"PC{i+1}" for i in range(len(eigvals))]
    )

    return eigvals, eigvecs, scores_df


# ============================================================
# 2) PARTEA A: BUGET LOCAL (Buget.csv + LocPopulation.csv)
# ============================================================

# --- Citire fișiere
buget = pd.read_csv(os.path.join(INPUT_DIR, "Buget.csv"))
locpop = pd.read_csv(os.path.join(INPUT_DIR, "LocPopulation.csv"))

# Ideea join-ului:
# - Buget.csv are Siruta + City + V1..V5 + C1..C8 (valori totale)
# - LocPopulation.csv are Siruta + Population + County (și uneori City)
# Noi avem nevoie de Population și County pentru calcule.
# De aceea facem merge pe Siruta.

# Păstrăm coloanele utile din locpop (ca să evităm suprapuneri inutile)
# Dacă City există și în locpop, nu ne încurcă (dar buget are deja City).
df = buget.merge(
    locpop[["Siruta", "Population", "County"]],
    on="Siruta",
    how="left"
)

# Curățare minimală:
# - dacă populația lipsește, nu putem calcula pe locuitor => eliminăm acele rânduri
df = df.dropna(subset=["Population"]).copy()

# Listele de categorii (din enunț)
income_cols = ["V1", "V2", "V3", "V4", "V5"]
expense_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

# ------------------------------------------------------------
# CERINȚA 1:
# "Save in Requirement_1.csv incomes and expenses per inhabitant by category"
# Pentru fiecare localitate -> Siruta + V1..V5 per capita + C1..C8 per capita
# ------------------------------------------------------------
for col in income_cols + expense_cols:
    df[col + "_per_inhab"] = per_capita(df[col], df["Population"])

req1_cols = ["Siruta"] + [c + "_per_inhab" for c in income_cols + expense_cols]
requirement_1 = df[req1_cols].copy()
requirement_1.to_csv(os.path.join(OUTPUT_DIR, "Requirement_1.csv"), index=False)

# ------------------------------------------------------------
# CERINȚA 2:
# "percentages of expenses for each category in total expenses at county level"
# Pași:
# 1) Agregăm cheltuielile pe județ (sumă C1..C8)
# 2) Transformăm în procente din total cheltuieli
# 3) Salvăm County + procente C1..C8
# ------------------------------------------------------------
county_sum_exp = df.groupby("County")[expense_cols].sum()
county_pct_exp = percent_structure(county_sum_exp)

requirement_2 = county_pct_exp.reset_index()  # aducem County din index în coloană
requirement_2.to_csv(os.path.join(OUTPUT_DIR, "Requirement_2.csv"), index=False)

# ============================================================
# 3) PARTEA B: PCA pe DataSet_25.csv
# ============================================================

ds = pd.read_csv(os.path.join(INPUT_DIR, "DataSet_25.csv"))

# În seturile DSAD, de obicei prima coloană e identificatorul (ex: Country)
# Îl punem ca index ca să nu intre în calculele numerice.
ds = ds.set_index(ds.columns[0])

# Păstrăm doar variabilele numerice (cele 8 din enunț)
X = ds.select_dtypes(include=[np.number]).copy()

# ------------------------------------------------------------
# Standardizare
# ------------------------------------------------------------
Xstd = zscore_standardize(X)

# ------------------------------------------------------------
# CERINȚA 3:
# "StdCov.csv = matricea de var-cov pentru variabilele standardizate"
# ------------------------------------------------------------
Cov = covariance_matrix(Xstd)
Cov.to_csv(os.path.join(OUTPUT_DIR, "StdCov.csv"))

# ------------------------------------------------------------
# CERINȚA 4:
# "PCA și scorurile (standardised principal components) -> Scores.csv"
# ------------------------------------------------------------
eigvals, eigvecs, Scores = pca_scores_from_cov(Xstd, Cov)
Scores.to_csv(os.path.join(OUTPUT_DIR, "Scores.csv"))

# ------------------------------------------------------------
# CERINȚA 5:
# "Scree plot + linie orizontală la varianța = 1"
# (criteriul Kaiser: păstrezi componente cu eigval > 1)
# ------------------------------------------------------------
x_pc = np.arange(1, len(eigvals) + 1)

plt.figure()
plt.plot(x_pc, eigvals, marker="o")
plt.axhline(y=1, linestyle="--")
plt.xticks(x_pc)
plt.xlabel("Componenta principală (PC)")
plt.ylabel("Valoare proprie (varianță explicată)")
plt.title("Scree plot (PCA) + criteriul Kaiser (linie la 1)")
plt.tight_layout()

# dacă vrei și salvare:
plt.savefig(os.path.join(OUTPUT_DIR, "ScreePlot.png"), dpi=200)

plt.show()


# ============================================================
# 4) Mesaj scurt de final (util când rulezi în consolă)
# ============================================================
print("Gata! Fișiere generate în dataOUT/: Requirement_1.csv, Requirement_2.csv, StdCov.csv, Scores.csv, ScreePlot.png")
