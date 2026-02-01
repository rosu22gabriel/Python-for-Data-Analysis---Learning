import pandas as pd
import numpy as np
import os

FOLDER_DATE = "data_in"


tari_est_europa = [
    "Romania", "Bulgaria", "Poland", "Hungary", "Czechia", "Slovakia",
    "Slovenia", "Croatia", "Estonia", "Latvia", "Lithuania"
]

# (fisier, nume_coloana_finala)
fisiere_indicatori = [
    ("pib_pe_cap_de_locuitor_nefiltrat.csv", "pib_pe_cap_de_locuitor"),
    ("speranta_de_viata_nefiltrat.csv", "speranta_de_viata"),
    ("populatie_urbana_nefiltrat.csv", "populatie_urbana_proc"),
    ("utilizatori_internet_nefiltrat.csv", "utilizatori_internet_proc"),
    ("cheltuieli_educatie_nefiltrat.csv", "cheltuieli_educatie_proc_PIB"),
]

def citire_indicator(cale_fisier):
    df = pd.read_csv(cale_fisier, skiprows=4)
    df = df[df["Country Name"].isin(tari_est_europa)].copy()

    coloane_ani = [c for c in df.columns if str(c).isdigit()]

    # curatare valori lipsa World Bank
    df[coloane_ani] = df[coloane_ani].replace(["..", "NA", "N/A", "", "null", "NULL"], np.nan)

    # numeric
    for c in coloane_ani:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # long
    df_lung = df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=coloane_ani,
        var_name="An",
        value_name="Valoare"
    )
    df_lung["An"] = pd.to_numeric(df_lung["An"], errors="coerce").astype("Int64")
    return df_lung

def ultimii_ani_disponibili(df_lung):
    # ultimul an care are macar o valoare (pentru tarile filtrate)
    ani = df_lung.dropna(subset=["Valoare"])["An"].dropna().unique()
    ani = [int(a) for a in ani]
    return sorted(ani)

# 1) citim fiecare indicator si pastram separat
indicator_long = {}
ani_posibili = set()

for (fis, col_finala) in fisiere_indicatori:
    cale = os.path.join(FOLDER_DATE, fis)
    if not os.path.exists(cale):
        raise FileNotFoundError(f"Nu gasesc fisierul: {cale}")

    df_lung = citire_indicator(cale)
    indicator_long[col_finala] = df_lung

    # ani posibili pentru indicator
    ani = ultimii_ani_disponibili(df_lung)
    ani_posibili.update(ani)

ani_posibili = sorted(list(ani_posibili))

# 2) alegem un an "comun bun": cel mai recent an care maximizeaza completitudinea
# completitudine = cate tari au valori la TOTI indicatorii
def scor_an(an):
    # pornim cu lista tarilor
    baza = pd.DataFrame({"Country Name": tari_est_europa})

    for col_finala, df_lung in indicator_long.items():
        sub = df_lung[df_lung["An"] == an][["Country Name", "Valoare"]].copy()
        sub = sub.rename(columns={"Valoare": col_finala})
        baza = baza.merge(sub, on="Country Name", how="left")

    # cate tari au toate valorile (fara NaN)
    complete = baza.dropna()
    return complete.shape[0], baza

best_an = None
best_complete = -1
best_baza = None

for an in sorted(ani_posibili, reverse=True):
    complete_cnt, baza = scor_an(an)
    if complete_cnt > best_complete:
        best_complete = complete_cnt
        best_an = an
        best_baza = baza
    # daca avem completitudine foarte buna, ne oprim devreme
    if complete_cnt >= int(0.8 * len(tari_est_europa)):  # 80% tari complete
        break

# 3) construim tabel final pentru anul ales
df_final = best_baza.copy()

# optional: elimina tari care au prea multe lipsuri (ex: lipsesc 3+ indicatori)
# aici pastram tot, ca sa nu pierzi tari.
# df_final = df_final[df_final.isna().sum(axis=1) <= 2]

# 4) export CSV in data_in
out_path = os.path.join(FOLDER_DATE, "dataset_cluster_est_europa.csv")
df_final.to_csv(out_path, index=False)

print("Setul de date final a fost creat cu succes.")
print("An ales:", best_an)
print("Tari complete (toate cele 5 coloane):", best_complete, "din", len(tari_est_europa))
print("Fisier salvat in:", out_path)
