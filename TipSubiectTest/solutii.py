import numpy as np
import pandas as pd

def pick_col(df: pd.DataFrame, candidates):
    """
    Alege prima coloana existenta dintr o lista de variante posibile.
    In subiecte, uneori coloana poate fi scrisa usor diferit (ex: 'Judet' vs 'County')
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None

def per_1000(indicator: pd.Series, population: pd.Series) -> pd.Series:
    """
    Calculeaza rata la 1000 locuitori:
        rata = indicator / populatie * 1000

    Protectie:
        - daca populatia e 0 sau lipsa, impartirea nu are sens => rezultatul devine NaN
    """
    pop = population.replace(0, np.nan)
    return indicator / pop * 1000

def coef_variatie(x: pd.Series) -> float:
    """
    Coeficient de variatie (CV) = abatere standard / medie.

    Interpretare:
    - CV mare => valorile variaza mult intre localitatile judetului (dispersie mare)
    - CV mic => valorile sunt relativ unfirme in judet

    Nota:
    - daca media e 0, CV nu e definit => intoarcem NaN
    """
    mean = x.mean()
    if mean == 0 or pd.isna(mean):
        return np.nan
    # ddof=0 => abatere standard de populatie
    std = x.std(ddof=0)
    return std / mean

# 2) Citire Date
misc = pd.read_csv('./dataIN/MiscareaNatLoc.csv')
pop = pd.read_csv('./dataIN/PopulatieLocalitati.csv')

# Coloane asteptate in MiscareaNatLoc.csv (din enunt):
# Siruta, Localitate, Casatorii, Decedati, DecedatiSub1An, Divorturi, NascutiMorti, NascutiVii

# In PopulatieLocalitati.csv: populatia + indicativ judet pentru fiecare localitate

# Detectam robust coloanele cheie din fisierul cu populatie
siruta_pop = pick_col(pop, ["Siruta", "siruta", "SIRUTA"])
judet_col = pick_col(pop, ["Judet", "judet", "County", "county", "CodJudet","CountyCode"])
pop_col = pick_col(pop, ["Populatie", "populatie", "Population", "population", "POP"])

if siruta_pop is None or judet_col is None or pop_col is None:
    raise ValueError(
        "Nu am gasit coloanele necesare in PopulatieLocalitati.csv.\n"
        f"Coloane gasite: {list(pop.columns)}\n"
        "Am nevoie de ceva echivalent cu Siruta / Judet / Populatie"
    )

# Facem merge ca sa aducem in acelasi tabel:
# - indicatori (din misc)
# - populatia + judetul (din pop)
df = misc.merge(
    pop[[siruta_pop, judet_col, pop_col]],
    left_on="Siruta", #Siruta din MiscareaNatLoc
    right_on=siruta_pop, # Siruta din PopulatieLocalitati
    how="left"
)

# Daca exista randuri fara populatie / judet, nu putem face rate sau agregare corecta
df = df.dropna(subset=[judet_col, pop_col]).copy()

# 3) Cerinta 1
# "Localitatile unde numarul de divorturi > numarul de casatorii"
# Salvam: Siruta, Localitate, Divorturi, Casatorii
cerinta1 = df.loc[df["Divorturi"] > df["Casatorii"], ["Siruta", "Localitate", "Divorturi", "Casatorii"]].copy()
cerinta1.to_csv('./dataOUT/Cerinta1.csv', index=False)

# 4) Cerinta 2
# 'Nascuti vii la 1000 locuitori, la nivel de localitate"
# Salvam: Siruta, Localitate, NascutiVii_1000loc
# si sortam descrescator dupa aceasta rata.

df["NascutiVii_1000loc"] = per_1000(df["NascutiVii"], df[pop_col])

cerinta2 = df[["Siruta", "Localitate", "NascutiVii_1000loc"]].copy()
cerinta2 = cerinta2.sort_values(by="NascutiVii_1000loc", ascending=False)
cerinta2.to_csv('./dataOUT/Cerinta2.csv', index=False)

# 5) Cerinta 3
# "Indicatorii la nivel de judet" (agregare pe judet)
# Practic: adunam (sum) valorile pe localitati din acelasi judet.
# Salvam: Judet + indicatorii
indicators = ["Casatorii", "Decedati", "DecedatiSub1An", "Divorturi", "NascutiMorti", "NascutiVii"]

cerinta3 = (
    df.groupby(judet_col, as_index=False)[indicators]
        .sum()
        .rename(columns={judet_col: "Judet"})
)

cerinta3.to_csv('./dataOUT/Cerinta3.csv', index=False)

# 6) Cerinta 4
# "Coeficientii de variatie ai indicatorilor la nivel de judet"
# Pentru fiecare judet si fiecare indicator:
# CV = std / mean
# unde std si mean se calculeaza pe localitatile din judet
# Salvam: Judet + CV pentru fiecare indicator
# groupby pe judet, apoi aplicam functia coef_variatie pe fiecare coloana indicator

cerinta4 = df.groupby(judet_col)[indicators].agg(coef_variatie)

# Punem judet ca o coloana normala
cerinta4 = cerinta4.reset_index().rename(columns={judet_col: "Judet"})

cerinta4.to_csv('./dataOUT/Cerinta4.csv', index=False)




