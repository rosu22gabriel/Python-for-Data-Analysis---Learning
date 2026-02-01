import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SETĂRI GENERALE
# ============================================================
INPUT_DIR = "dataIN"
OUTPUT_DIR = "dataOUT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FUNCȚIE DE STANDARDIZARE Z-SCORE
# ============================================================
def standardizare_zscore(X):
    """
    Standardizare Z-score:
    - scădem media
    - împărțim la deviația standard
    Fiecare variabilă va avea:
      media = 0, deviația standard = 1
    """
    return (X - X.mean()) / X.std(ddof=0)


# ============================================================
# PARTEA 1 – RATE DEMOGRAFICE
# ============================================================

nat = pd.read_csv(os.path.join(INPUT_DIR, "NatLocMovement.csv"))
pop = pd.read_csv(os.path.join(INPUT_DIR, "PopulationLoc.csv"))

# luăm explicit City din nat și doar populația + county din pop
df = nat[["Siruta","City","Marriages","Deceased","DeceasedUnder1Year",
          "Divorces","StillBirths","LiveBirths"]].merge(
    pop[["Siruta","Population","County"]],
    on="Siruta",
    how="left"
)


# ------------------------------------------------------------
# CERINȚA 1 – Rata mortalității infantile (%)
# ------------------------------------------------------------
df["InfantMortalityRate"] = (
    df["DeceasedUnder1Year"] / df["LiveBirths"]
) * 100

req1 = df[["Siruta", "InfantMortalityRate"]].copy()
req1["City"] = nat["City"].values
req1 = req1[["Siruta", "City", "InfantMortalityRate"]]

req1.to_csv(os.path.join(OUTPUT_DIR, "Requirement_1.csv"), index=False)

# ------------------------------------------------------------
# CERINȚA 2 – Rate maxime pe județ (la 1000 locuitori)
# ------------------------------------------------------------
indicators = [
    "Marriages", "Deceased", "DeceasedUnder1Year",
    "Divorces", "StillBirths", "LiveBirths"
]

for col in indicators:
    df[col + "_rate"] = (df[col] / df["Population"]) * 1000

rows = []

for county in df["County"].unique():
    sub = df[df["County"] == county]
    row = {"County": county}

    for col in indicators:
        idx = sub[col + "_rate"].idxmax()
        row[col] = sub.loc[idx, "City"]

    rows.append(row)

req2 = pd.DataFrame(rows)
req2.to_csv(os.path.join(OUTPUT_DIR, "Requirement_2.csv"), index=False)

# ============================================================
# PARTEA 2 – ANALIZA ÎN COMPONENTE PRINCIPALE (ACP)
# ============================================================

ds = pd.read_csv(os.path.join(INPUT_DIR, "DataSet_83.csv"))
ds = ds.set_index(ds.columns[0])  # numele țării ca index

X = ds.select_dtypes(include=[np.number])

# ------------------------------------------------------------
# STANDARDIZARE
# ------------------------------------------------------------
Xstd = standardizare_zscore(X)

# ------------------------------------------------------------
# CERINȚA 3 – Matricea de variață-covarianță
# ------------------------------------------------------------
Cov = np.cov(Xstd.T)
pd.DataFrame(Cov, index=X.columns, columns=X.columns) \
  .to_csv(os.path.join(OUTPUT_DIR, "StdCov.csv"))

# ------------------------------------------------------------
# CERINȚA 4 – Componente principale
# ------------------------------------------------------------
# valori și vectori proprii
eigvals, eigvecs = np.linalg.eig(Cov)

# sortare descrescătoare
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# scoruri ACP
PrinComp = Xstd.values @ eigvecs
PrinComp_df = pd.DataFrame(
    PrinComp,
    index=X.index,
    columns=[f"PC{i+1}" for i in range(len(eigvals))]
)

PrinComp_df.to_csv(os.path.join(OUTPUT_DIR, "PrinComp.csv"))

# CERINTA 5 - Varianta explicata

plt.figure()
plt.plot(np.arange(1, len(eigvals)+1), eigvals, marker="o")
plt.axhline(y=1, linestyle="--")
plt.xlabel("Componenta principală")
plt.ylabel("Varianță explicată")
plt.title("Scree plot")
plt.show()


# CERINTA 6 - Loadings + cercul corelatiilor
loadings = eigvecs * np.sqrt(eigvals)
plt.figure(figsize=(6,6))
plt.axhline(0)
plt.axvline(0)

for i, var in enumerate(X.columns):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
              head_width=0.03, length_includes_head=True)
    plt.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, var)

circle = plt.Circle((0, 0), 1, fill=False)
plt.gca().add_artist(circle)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Cercul corelatiilor")
plt.axis("equal")
plt.show()

