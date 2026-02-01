import os
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


# ============================================================
# SETĂRI GENERALE (ca în proiectele/seminariile DSAD)
# ============================================================
INPUT_DIR = "dataIN"
OUTPUT_DIR = "dataOUT"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pd.set_option("display.max_columns", None)
np.set_printoptions(precision=6, suppress=True)


# ============================================================
# FUNCȚII AJUTĂTOARE (stil seminar)
# ============================================================
def zscore_standardize(df_numeric: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizare Z-score pe coloane:
        z = (x - media) / dev_std
    Observație:
      - dacă o coloană are dev_std = 0 (toate valori identice), evităm împărțirea la 0.
    """
    means = df_numeric.mean(axis=0)
    stds = df_numeric.std(axis=0, ddof=0)  # ddof=0 = "populație" (consistent cu multe seminarii)
    stds_replaced = stds.replace(0, 1)     # protecție: dacă std=0 -> nu standardizăm, rămâne 0 după centrare

    return (df_numeric - means) / stds_replaced


def rate_per_1000(indicator: pd.Series, population: pd.Series) -> pd.Series:
    """
    Rată la 1000 locuitori:
        indicator / populație * 1000
    Protecție pentru populație 0 sau lipsă.
    """
    pop = population.replace(0, np.nan)
    return (indicator / pop) * 1000


# ============================================================
# PARTEA 1-2: NatLocMovement + PopulationLoc
# ============================================================
# --- citire date
nat_path = os.path.join(INPUT_DIR, "NatLocMovement.csv")
pop_path = os.path.join(INPUT_DIR, "PopulationLoc.csv")

nat = pd.read_csv(nat_path)
pop = pd.read_csv(pop_path)

# Așteptări (din enunț):
# NatLocMovement.csv: Siruta, City, Marriages, Deceased, DeceasedUnder1Year, Divorces, StillBirths, LiveBirths
# PopulationLoc.csv: populația pe localități + cod județ (county) pentru fiecare localitate
#
# IMPORTANT: Nu știm exact numele coloanelor din PopulationLoc.csv în poza ta.
# Cel mai des în seminarii apare ceva de tip: Siruta, County, Population (sau Pop).
# Mai jos facem o detecție robustă pe nume uzuale.

# --- identificare coloane cheie în PopulationLoc.csv
# (dacă ai alte nume exacte, le înlocuiești aici)
siruta_col_candidates = [c for c in pop.columns if c.lower() in ["siruta", "sirutacode", "siruta_code"]]
county_col_candidates = [c for c in pop.columns if c.lower() in ["county", "judet", "countycode", "cod_judet", "codjudet"]]
pop_col_candidates = [c for c in pop.columns if c.lower() in ["population", "pop", "populatie", "nr_locuitori"]]

if not siruta_col_candidates or not county_col_candidates or not pop_col_candidates:
    raise ValueError(
        "Nu am putut detecta automat coloanele din PopulationLoc.csv.\n"
        f"Coloane găsite: {list(pop.columns)}\n"
        "Asigură-te că ai coloane de tip Siruta / County / Population (sau adaptează mapping-ul)."
    )

SIRUTA = siruta_col_candidates[0]
COUNTY = county_col_candidates[0]
POP = pop_col_candidates[0]

# --- join NatLocMovement cu PopulationLoc după Siruta (cheie de localitate)
df = nat.merge(pop[[SIRUTA, COUNTY, POP]], left_on="Siruta", right_on=SIRUTA, how="left")

# Dacă există localități fără populație, nu poți calcula rate -> le marcăm / eliminăm din calcule
df_valid = df.dropna(subset=[POP]).copy()

# ------------------------------------------------------------
# CERINȚA 1:
# "Rata sporului natural la nivel de județ (county)" în Request_1.csv
# spor natural = rata nașterilor (LiveBirths/1000) - rata deceselor (Deceased/1000)
#
# FOARTE IMPORTANT:
# Pentru județ:
#   - NU faci media ratelor pe localități (ar fi greșit statistic).
#   - corect: agregi întâi indicatorii și populația pe județ, apoi calculezi rata.
# ------------------------------------------------------------
county_agg = (
    df_valid
    .groupby(COUNTY, as_index=False)
    .agg(
        Pop_total=(POP, "sum"),
        LiveBirths_total=("LiveBirths", "sum"),
        Deceased_total=("Deceased", "sum")
    )
)

county_agg["BirthRate_per_1000"] = rate_per_1000(county_agg["LiveBirths_total"], county_agg["Pop_total"])
county_agg["DeathRate_per_1000"] = rate_per_1000(county_agg["Deceased_total"], county_agg["Pop_total"])

# spor natural (rata spor natural) = birth rate - death rate
county_agg["NaturalIncreaseRate_per_1000"] = county_agg["BirthRate_per_1000"] - county_agg["DeathRate_per_1000"]

request_1 = county_agg[[COUNTY, "NaturalIncreaseRate_per_1000"]].copy()
request_1.columns = ["County", "NaturalIncreaseRate_per_1000"]

request_1.to_csv(os.path.join(OUTPUT_DIR, "Request_1.csv"), index=False)


# ------------------------------------------------------------
# CERINȚA 2:
# "Pentru fiecare județ, localitățile unde ratele sunt maxime"
# Indicatori ceruți (din enunț):
# Marriages, Deceased, DeceasedUnder1Year, Divorces, StillBirths, LiveBirths
#
# Pași:
#  1) calculezi rate per localitate (indicator/pop*1000)
#  2) pe fiecare județ și indicator alegi localitatea cu rata maximă
#  3) afișezi County + numele localității maxime pentru fiecare indicator
# ------------------------------------------------------------
indicators = ["Marriages", "Deceased", "DeceasedUnder1Year", "Divorces", "StillBirths", "LiveBirths"]

# calculăm ratele per 1000 pe localitate
for col in indicators:
    df_valid[col + "_rate"] = rate_per_1000(df_valid[col], df_valid[POP])

# pentru fiecare județ, pentru fiecare indicator, luăm City-ul cu rata maximă
result_rows = []
counties = df_valid[COUNTY].dropna().unique()

for c in sorted(counties):
    sub = df_valid[df_valid[COUNTY] == c].copy()

    row = {"County": c}
    for col in indicators:
        rate_col = col + "_rate"

        # idx cu rata maximă (ignorând NaN)
        idx_max = sub[rate_col].idxmax()
        city_max = sub.loc[idx_max, "City"] if pd.notna(idx_max) else np.nan

        # dacă vrei să tratezi egalități: poți extrage toate localitățile cu max și să le unești
        # max_val = sub[rate_col].max()
        # city_max = ", ".join(sorted(sub.loc[sub[rate_col] == max_val, "City"].astype(str).unique()))

        row[col] = city_max

    result_rows.append(row)

request_2 = pd.DataFrame(result_rows, columns=["County"] + indicators)
request_2.to_csv(os.path.join(OUTPUT_DIR, "Request_2.csv"), index=False)


# ============================================================
# PARTEA 3-5: DataSet_43.csv (35 țări) - clustering ierarhic Ward
# ============================================================
ds_path = os.path.join(INPUT_DIR, "DataSet_43.csv")
ds = pd.read_csv(ds_path)

# De obicei, în seturile de la seminar:
# - prima coloană e numele țării (string)
# - restul sunt variabile numerice (POP, PIB, CHS_PUB, CHS_PRIV, RM, RN, SPV)
#
# Facem detectare: nume țară = prima coloană non-numerică (sau chiar prima coloană)
country_col = ds.columns[0]  # presupunere rezonabilă pentru seturi tipice seminar
ds = ds.set_index(country_col)

# păstrăm doar numerice
X = ds.select_dtypes(include=[np.number]).copy()

# --- CERINȚA 3: standardizare + ierarhie ward/euclidean
Xstd = zscore_standardize(X)

# salvare Xstd.csv
Xstd.to_csv(os.path.join(OUTPUT_DIR, "Xstd.csv"))

# matricea ierarhiei (linkage)
# Z are coloane: [cluster1, cluster2, distanta, nr_obs_in_cluster_nou]
Z = linkage(Xstd.values, method="ward", metric="euclidean")

# "ierarhia e printată în consolă" (cerință)
print("=== Matricea ierarhiei (linkage) ===")
print(Z)

# --- CERINȚA 4: pragul (threshold) și joncțiunea cu diferența maximă
# Distantele de agregare sunt Z[:, 2]
dist = Z[:, 2]
diff = np.diff(dist)

# poziția unde crește cel mai mult distanța între două agregări consecutive
k = np.argmax(diff)  # index în diff; între dist[k] și dist[k+1] e saltul maxim

# pragul: de obicei îl alegem "la mijloc" între cele două niveluri,
# ca să tăiem dendrograma în "spațiul" dintre ele
threshold = (dist[k] + dist[k + 1]) / 2

# "joncțiunea" = pasul/îmbinarea (merge) la care apare saltul (poți raporta k+1 ca pas 1-indexat)
junction = k + 1  # 1-indexat ar fi (k+1)

print("\n=== Stabilitate maximă (diferență maximă între distanțe) ===")
print(f"Max diff = {diff[k]:.6f} între dist[{k}]={dist[k]:.6f} și dist[{k+1}]={dist[k+1]:.6f}")
print(f"Threshold (prag) ales = {threshold:.6f}")
print(f"Junction (pas/îmbinare) = {junction} (începând de la 1)")

# --- CERINȚA 5: dendrogramă + linie orizontală la threshold + rotație 45°
plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    labels=Xstd.index.astype(str).tolist(),
    leaf_rotation=45
)
plt.axhline(y=threshold, linewidth=2)
plt.title("Dendrogramă (Ward + Euclidian) cu pragul de tăiere")
plt.ylabel("Distanța de agregare")
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "dendrogram_threshold.png"), dpi=200)
plt.show()
