"""
Seminar 10 – PCA (ACP) + Analiză factorială + Hărți (TOTUL într-un singur fișier)
================================================================================

Acest script combină într-un singur fișier:
- functii.py (nan_replace_df, acp, tabelare_varianta, salvare_ndarray, etc.)
- grafice.py (plot_varianta, corelograma, scatterplot, plot_harta)
- main.py (PCA/ACP complet)
- main_fa.py (Factor Analysis complet, cu Bartlett + KMO)

------------------------------------------------------------
STRUCTURA proiectului (așteptată):
Seminar10_1089/
  main_single.py   (ăsta)
  data_in/
    Territorial2022/
      Territorial_2022.csv
  RO_NUTS2/
    Ro.shp + restul fișierelor shapefile
  graphics/         (se creează dacă nu există)
  data_out/         (se creează dacă nu există)         -> PCA
  data_out_fa/      (se creează dacă nu există)         -> FA

------------------------------------------------------------
CE PRODUCE PCA (ACP):
- data_out/R.csv        : matricea de covarianță/correlație (R_v)
- data_out/Varianta.csv : varianța componentelor + procente + cumul
- data_out/C.csv        : scoruri pe componente (C)
- data_out/S.csv        : scoruri standardizate (S = C / sqrt(alpha))
- data_out/R_XC.csv     : corelații variabile–componente
- data_out/Cosin.csv    : cosinusuri (calitatea reprezentării observațiilor)
- data_out/Contrib.csv  : contribuții observații la componente
- data_out/Comm.csv     : comunalități (cât de bine explică componentele variabilele)
- în folderul graphics/: corelograme, plot varianta, scatter-uri, hărți scoruri etc.

------------------------------------------------------------
CE PRODUCE FA (Analiză factorială):
- data_out_fa/kmo.csv                : KMO pe variabile + total
- data_out_fa/Varianta.csv           : varianța factorilor comuni
- data_out_fa/Varianta_specifica.csv : psi (unicitate / varianta specifică)
- data_out_fa/l.csv                  : încărcături factoriale (loadings)
- data_out_fa/f.csv                  : scoruri pe factori
- în folderul graphics/: corelograme, plot varianta, scatter-uri, hărți scoruri etc.

------------------------------------------------------------
DEPENDENȚE:
pip install numpy pandas matplotlib seaborn scikit-learn geopandas factor_analyzer

ATENȚIE:
- geopandas + shapefile: asigură-te că ai instalate corect (uneori pe Windows e mai sensibil).
- folderul "graphics/" trebuie să existe sau îl creăm noi.

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pentru corelograme
from seaborn import heatmap

# Pentru hartă (shapefile)
from geopandas import GeoDataFrame

# Pentru analiză factorială
import factor_analyzer as fa


# =============================================================================
# 0) SETĂRI DE AFIȘARE (ca în proiect)
# =============================================================================
np.set_printoptions(precision=3, threshold=sys.maxsize, suppress=True)
pd.set_option("display.max_columns", None)


# =============================================================================
# 1) FUNCȚII UTILITARE (din functii.py) – explicate “pe bune”
# =============================================================================

from pandas.api.types import is_numeric_dtype


def nan_replace_df(t: pd.DataFrame):
    """
    Înlocuiește valorile lipsă (NaN) din DataFrame, "in place".

    De ce e necesar:
    - PCA/FA lucrează cu calcule de covarianță/correlație și nu acceptă NaN.
    - Dacă există NaN, rezultatele pot pica sau pot deveni invalide.

    Regula folosită:
    - Dacă o coloană e numerică -> înlocuim NaN cu media coloanei.
    - Dacă o coloană NU e numerică -> înlocuim NaN cu moda (cea mai frecventă valoare).
    """
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


def acp(x: np.ndarray, scal: bool = True, ddof: int = 0):
    """
    Implementare manuală PCA/ACP (ca în proiect), pe baza matricii de covarianță/correlație.

    Parametri:
    - x: matricea datelor (n observații x m variabile)
    - scal: dacă True, standardizăm variabilele (z-score) -> PCA pe CORRELAȚII
            dacă False, doar centrare -> PCA pe COVARIANȚĂ
    - ddof: grade de libertate pentru std/cov (în proiect e 0)

    Pași:
    1) Centrare: x_ = x - media pe coloană
    2) (opțional) Standardizare: împărțim la deviația standard pe coloană
    3) Matrice R_v = (1/(n-ddof)) * x_^T x_
       - dacă am standardizat -> R_v ~ matrice de corelații
       - dacă nu -> matrice de covarianțe
    4) Descompunere în valori proprii:
       R_v * a = alpha * a
       - alpha = valorile proprii (varianța pe componente)
       - a = vectorii proprii (axe / încărcături)
    5) Sortăm descrescător după alpha.

    Returnează:
    - x_     : datele centrate (și eventual standardizate)
    - r_v    : matricea R_v
    - alpha  : valori proprii sortate descrescător
    - a      : vectori proprii corespunzători (coloane)
    """
    n, m = x.shape
    x_ = x - np.mean(x, axis=0)

    if scal:
        # Standardizare la deviația standard => toate variabilele au aceeași scară
        x_ = x_ / np.std(x, axis=0, ddof=ddof)

    # Matricea de covarianță/correlație
    r_v = (1 / (n - ddof)) * x_.T @ x_

    # Eigen-decomposition
    valp, vecp = np.linalg.eig(r_v)

    # Sortăm valorile proprii descrescător
    k = np.flip(np.argsort(valp))
    alpha = valp[k]
    a = vecp[:, k]

    return x_, r_v, alpha, a


def tabelare_varianta(alpha: np.ndarray):
    """
    Construiește tabelul cu varianță, varianță cumulată, procent, procent cumulat
    pentru componentele PCA.

    alpha[i] = varianța componentei Ci.

    Return: DataFrame indexat cu C1, C2, ...
    """
    m = len(alpha)
    procent = alpha * 100 / sum(alpha)

    t = pd.DataFrame(
        data={
            "Varianta": alpha,
            "Varianta cumulata": np.cumsum(alpha),
            "Procent varianta": procent,
            "Procent cumulat": np.cumsum(procent),
        },
        index=["C" + str(i + 1) for i in range(m)]
    )
    return t


def tabelare_varianta_factori(varianta):
    """
    Pentru FA (factor_analyzer) varianta vine ca:
      varianta[0] = varianța factorilor comuni (eigenvalues pe factori)
      varianta[1] = proporția de varianță explicată de fiecare factor
      varianta[2] = proporția cumulată

    Construim tabel cu F1, F2, ... (similar PCA).
    """
    alpha = varianta[0]
    t = pd.DataFrame(
        data={
            "Varianta": alpha,
            "Varianta cumulata": np.cumsum(alpha),
            "Procent varianta": varianta[1] * 100,
            "Procent cumulat": varianta[2] * 100,
        },
        index=["F" + str(i + 1) for i in range(len(alpha))]
    )
    return t


def salvare_ndarray(x: np.ndarray,
                    nume_linii,
                    nume_coloane,
                    nume_index=None,
                    nume_fisier_output=None):
    """
    Transformă un numpy array într-un DataFrame cu:
    - index = nume_linii
    - columns = nume_coloane
    și (opțional) îl salvează în CSV.

    Foarte util ca să păstrezi:
    - scoruri, corelații, cosinusuri etc. în fișiere “curate”.
    """
    temp = pd.DataFrame(x, index=nume_linii, columns=nume_coloane)
    temp.index.name = nume_index
    if nume_fisier_output is not None:
        temp.to_csv(nume_fisier_output)
    return temp


# =============================================================================
# 2) FUNCȚII DE GRAFICE (din grafice.py) – explicate
# =============================================================================

def plot_varianta(alpha: np.ndarray,
                  procent_minimal=80,
                  scal=True,
                  eticheta_x="Componenta"):
    """
    Desenează:
    - linia valorilor alpha (varianța componentelor/factorilor)
    - criterii ca să alegi nr. de componente/factori:

    (1) Kaiser (doar dacă scal=True):
        - păstrezi componentele cu alpha > 1
    (2) Procent minimal:
        - păstrezi până când procentul cumulat trece peste procent_minimal (ex: 80%)
    (3) Cattell / Elbow:
        - caută prima “curbură” (heuristică)

    Returnează (k1, k2, k3):
    - k1 = nr. de componente după Kaiser (sau None)
    - k2 = nr. minim pt procent_minimal
    - k3 = nr. după elbow (sau None)

    Important:
    - În proiect, se folosește min dintre valorile existente (nu None).
    """
    m = len(alpha)
    x = np.arange(1, m + 1)

    f = plt.figure(figsize=(8, 5))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title("Plot varianta", color="b", fontsize=18)
    ax.set_xlabel(eticheta_x, fontsize=12)
    ax.set_ylabel("Varianta", fontsize=12)
    ax.set_xticks(x)

    ax.plot(x, alpha)
    ax.scatter(x, alpha, c="r", alpha=0.5)

    # (1) Kaiser: alpha > 1
    k1 = None
    if scal:
        k1 = len(np.where(alpha > 1)[0])
        ax.axhline(1, c="g", label="Criteriul Kaiser")

    # (2) Procent minimal cumul
    procent_cumulat = np.cumsum(alpha * 100 / sum(alpha))
    k2 = np.where(procent_cumulat > procent_minimal)[0][0] + 1
    ax.axvline(k2, c="c", label=f"Procent minimal ({procent_minimal}%)")

    # (3) Elbow (Cattell) – heuristica din proiect
    k3 = None
    eps = alpha[:m - 1] - alpha[1:]
    sigma = eps[:m - 2] - eps[1:]
    negative = sigma < 0
    if any(negative):
        k3 = np.where(negative)[0][0] + 2
        ax.axvline(k3, c="m", label="Criteriul Cattell (Elbow)")

    ax.legend()

    os.makedirs("graphics", exist_ok=True)
    plt.savefig("graphics/PlotVarianta.png")

    return k1, k2, k3


def corelograma(t: pd.DataFrame,
                titlu="Corelograma",
                vmin=-1,
                cmap="RdYlBu",
                annot=True,
                vmax=1):
    """
    Heatmap (corelogramă) pentru un tabel (de obicei corelații, cosinusuri, contribuții etc.)

    Parametri importanți:
    - vmin/vmax: intervalul culorilor (de ex -1..1 pentru corelații)
    - cmap: schema de culori
    - annot: dacă True, scrie numerele în celule (bun la tabele mici)
    """
    f = plt.figure(figsize=(8, 8))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu, color="b", fontsize=18)

    heatmap(t, vmin=vmin, vmax=vmax, cmap=cmap, annot=annot, ax=ax)

    os.makedirs("graphics", exist_ok=True)
    plt.savefig("graphics/" + titlu + ".png")


def scatterplot(t: pd.DataFrame,
                x="C1",
                y="C2",
                titlu="Plot scoruri",
                etichete=None,
                corelatii=False):
    """
    Scatter plot pentru:
    - scoruri (observații în spațiul componentelor: C1 vs C2 etc.)
    - corelații variabile–componente (cercul corelațiilor), dacă corelatii=True

    Dacă corelatii=True:
    - desenează cercul unitate (radius=1) + cercul 0.7 (ajută vizual)
    - interpretarea e tip "cercul corelațiilor" din PCA.

    etichete:
    - dacă e listă, scriem text lângă fiecare punct
      (ex: numele variabilelor sau numele județelor)
    """
    f = plt.figure(figsize=(9, 6))
    ax = f.add_subplot(1, 1, 1, aspect=1)
    ax.set_title(titlu, color="b", fontsize=18)

    ax.scatter(t[x], t[y], c="r", alpha=0.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if corelatii:
        pas = 0.05
        theta = np.arange(0, np.pi * 2 + pas, pas)
        ax.plot(np.cos(theta), np.sin(theta))                 # cerc unitate
        ax.plot(0.7 * np.cos(theta), 0.7 * np.sin(theta), c="g")  # cerc 0.7

    ax.axvline(0, c="k")
    ax.axhline(0, c="k")

    if etichete is not None:
        n = len(etichete)
        for i in range(n):
            # Folosim .iloc ca să accesăm elementul i în ordine
            ax.text(t[x].iloc[i], t[y].iloc[i], str(etichete[i]))

    os.makedirs("graphics", exist_ok=True)
    plt.savefig(f"graphics/{titlu}_{x}_{y}.png")


def plot_harta(gdf: GeoDataFrame,
               camp_legatura,
               t: pd.DataFrame,
               camp_harta,
               cmap="Reds",
               titlu="Harta scoruri"):
    """
    Desenează o hartă colorată după o coloană (camp_harta) din tabelul t.

    Ideea:
    - gdf = shapefile-ul (geometria regiunilor)
    - t   = tabelul cu scoruri (index = cod regiune/județ)
    - camp_legatura = coloana din shapefile care corespunde indexului din t (în proiect: "sj")

    Pași:
    1) facem merge între gdf și t (pe camp_legatura vs index)
    2) plotez geometria colorată după t[camp_harta]
    """
    f = plt.figure(figsize=(9, 8))
    ax = f.add_subplot(1, 1, 1, aspect=1)
    ax.set_title(titlu + " - " + camp_harta, color="b", fontsize=18)

    gdf_ = gdf.merge(t, left_on=camp_legatura, right_index=True)
    gdf_.plot(column=camp_harta, cmap=cmap, legend=True, ax=ax)

    os.makedirs("graphics", exist_ok=True)
    plt.savefig("graphics/" + titlu + "-" + camp_harta + ".png")


def show():
    """Afișează toate figurile matplotlib."""
    plt.show()


# =============================================================================
# 3) PCA / ACP – logica din main.py (explicată)
# =============================================================================

def run_pca():
    """
    Rulează complet PCA (ca main.py), cu aceiași pași:

    1) Citește datele și completează NaN.
    2) Alege variabilele observate (din coloana a 4-a încolo: [3:])
    3) ACP manual (acp): obține x_, R_v, alpha, a
    4) Salvează R și face corelogramă
    5) Varianta: tabel, plot varianta, alegere nr. componente
    6) Scoruri C și S, corelații variabile–componente
    7) Plot-uri: cerc corelații + scoruri
    8) Metrici: cosinusuri, contribuții, comunalități
    9) Hărți scoruri pe componente
    """
    os.makedirs("data_out", exist_ok=True)
    os.makedirs("graphics", exist_ok=True)

    # 1) citire dataset
    set_date = pd.read_csv("data_in/Teritorial2022/Teritorial_2022.csv", index_col=0)
    nan_replace_df(set_date)

    # 2) selectare variabile observate
    # În dataset-ul tău, primele 3 coloane sunt probabil identificatori / meta.
    variabile_observate = list(set_date)[3:]

    x = set_date[variabile_observate].values

    # 3) ACP manual: x_ (standardizat), R_v, alpha, a
    x_, r_v, alpha, a = acp(x)

    # 4) salvare R și corelogramă
    t_r = salvare_ndarray(
        r_v,
        variabile_observate,
        variabile_observate,
        nume_index="Indicatori",
        nume_fisier_output="data_out/R.csv"
    )
    corelograma(t_r, annot=len(variabile_observate) < 10)

    # 5) analiza varianței
    t_varianta = tabelare_varianta(alpha)
    t_varianta.round(3).to_csv("data_out/Varianta.csv")

    # Alegem nr. componente folosind criterii (Kaiser / procent / elbow)
    k = plot_varianta(alpha)

    # În proiect: nr_comp = min([v for v in k if v is not None])
    nr_comp = min([v for v in k if v is not None])

    # 6) scoruri + corelații variabile–componente
    # c = X * A (scoruri pe componente)
    c = x_ @ a

    # s = C / sqrt(alpha) (standardizare a scorurilor)
    s = c / np.sqrt(alpha)

    etichete_componente = list(t_varianta.index)

    t_c = salvare_ndarray(
        c,
        set_date.index,
        etichete_componente,
        nume_index=set_date.index.name,
        nume_fisier_output="data_out/C.csv"
    )

    t_s = salvare_ndarray(
        s,
        set_date.index,
        etichete_componente,
        nume_index=set_date.index.name,
        nume_fisier_output="data_out/S.csv"
    )

    # r_XC = corelația dintre variabilele X (standardizate) și scorurile C
    n, m = x.shape
    r_XC = np.corrcoef(x_, c, rowvar=False)[:m, m:]
    t_r_XC = salvare_ndarray(
        r_XC,
        variabile_observate,
        etichete_componente,
        nume_index="Indicatori",
        nume_fisier_output="data_out/R_XC.csv"
    )

    corelograma(t_r_XC, "Corelatii variabile-componente", annot=m < 10)

    # 7) plot “cercul corelațiilor” și plot scoruri
    # În proiect pornește de la i=2: (C2, C3, ...)
    # (Se presupune că pe axa X e mereu C1, iar pe y e Ci)
    for i in range(2, nr_comp + 1):
        # Cercul corelațiilor: variabile în plan (C1, Ci)
        scatterplot(
            t_r_XC,
            x="C1",
            y="C" + str(i),
            titlu="Corelatii factoriale",
            etichete=variabile_observate,
            corelatii=True
        )

    for i in range(2, nr_comp + 1):
        # Scoruri observații: (C1, Ci)
        scatterplot(
            t_c,
            x="C1",
            y="C" + str(i),
            titlu="Plot componente",
            etichete=set_date.index
        )
        # Scoruri standardizate
        scatterplot(
            t_s,
            x="C1",
            y="C" + str(i),
            titlu="Plot scoruri",
            etichete=set_date.index
        )

    # 8) metrici: cosinusuri, contribuții, comunalități

    # cosinus = (C^2) / suma(C^2 pe rând)
    c2 = c * c
    cosin = (c2.T / np.sum(c2, axis=1)).T
    t_cosin = salvare_ndarray(
        cosin,
        set_date.index,
        etichete_componente,
        nume_index=set_date.index.name,
        nume_fisier_output="data_out/Cosin.csv"
    )
    if n < 50:
        corelograma(t_cosin, "Cosinusuri", vmin=0, cmap="Reds", annot=m < 15)

    # contribuții = C^2 / suma(C^2 pe coloană) * 100
    contrib = c2 * 100 / np.sum(c2, axis=0)
    t_contrib = salvare_ndarray(
        contrib,
        set_date.index,
        etichete_componente,
        nume_index=set_date.index.name,
        nume_fisier_output="data_out/Contrib.csv"
    )
    if n < 50:
        corelograma(t_contrib, "Contributii", vmin=0, cmap="Reds", annot=m < 15, vmax=100)

    # comunalități (în PCA): sumă cumulată a r^2 pe componente
    r2 = r_XC * r_XC
    comm = np.cumsum(r2, axis=1)
    t_comm = salvare_ndarray(
        comm,
        variabile_observate,
        etichete_componente,
        nume_index="Indicatori",
        nume_fisier_output="data_out/Comm.csv"
    )
    corelograma(t_comm, "Comunalitati", vmin=0, cmap="Blues", annot=m < 15)

    # 9) Hărți scoruri pe componente
    gdf = GeoDataFrame.from_file("RO_NUTS2/Ro.shp")

    camp_legatura = "sj"  # coloana din shapefile ce se potrivește cu index-ul din set_date/t_c
    for i in range(1, nr_comp + 1):
        plot_harta(gdf, camp_legatura, t_c, "C" + str(i))

    print(f"[PCA] Gata. Am selectat nr_comp = {nr_comp} componente semnificative.")
    return nr_comp


# =============================================================================
# 4) Factor Analysis – logica din main_fa.py (explicată)
# =============================================================================

def run_fa():
    """
    Rulează complet Analiza Factorială (ca main_fa.py):

    1) Citește datele și completează NaN.
    2) Test Bartlett: verifică dacă există corelații suficiente (altfel FA nu are sens)
    3) Index KMO: măsoară “adecvarea” eșantionului / factorabilitatea
    4) Construiește model FA (cu rotație varimax sau fără)
    5) Analizează varianța: factori comuni, varianta specifică (psi), comunalități
    6) Alege nr. minim de factori (din plot varianta)
    7) Încărcături factoriale (loadings) + plot corelații
    8) Scoruri pe factori + plot scoruri
    9) Hărți scoruri pe factori
    """
    os.makedirs("data_out_fa", exist_ok=True)
    os.makedirs("graphics", exist_ok=True)

    set_date = pd.read_csv("data_in/Teritorial2022/Teritorial_2022.csv", index_col=0)
    nan_replace_df(set_date)

    variabile_observate = list(set_date)[3:]
    x = set_date[variabile_observate].values
    n, m = x.shape

    # 1) Factorabilitate: Bartlett
    # Ipoteză nulă: matricea de corelații = identitate (nu există corelații) => FA nu are sens.
    test_bartlett = fa.calculate_bartlett_sphericity(x)
    # test_bartlett = (statistică, p-value)
    if test_bartlett[1] > 0.001:
        print("[FA] Bartlett p-value > 0.001 => Nu există suficienți factori comuni. Oprim.")
        return None

    # 2) KMO
    # kmo = (kmo_per_variabilă, kmo_total)
    kmo = fa.calculate_kmo(x)

    t_kmo = pd.DataFrame(
        data={"KMO": np.append(kmo[0], kmo[1])},
        index=variabile_observate + ["Total"]
    )
    t_kmo.to_csv("data_out_fa/kmo.csv")
    corelograma(t_kmo, "Index KMO", vmin=0, cmap="Greens", annot=True, vmax=1)

    # 3) Construire model factorial
    # rotația varimax ajută interpretarea (încărcături mai “curate” pe factori)
    metoda_rotatie = "varimax"
    # model_fact = fa.FactorAnalyzer(m, rotation=metoda_rotatie) -> pornește cu m factori
    model_fact = fa.FactorAnalyzer(m, rotation=metoda_rotatie)
    model_fact.fit(x)

    # 4) Analiza varianței
    varianta = model_fact.get_factor_variance()
    t_varianta = tabelare_varianta_factori(varianta)
    t_varianta.round(3).to_csv("data_out_fa/Varianta.csv")

    # psi = unicitatea (varianta specifică fiecărei variabile)
    psi = model_fact.get_uniquenesses()
    t_psi = pd.Series(np.append(psi, sum(psi)), index=variabile_observate + ["Total"], name="Varianta")
    t_psi.to_csv("data_out_fa/Varianta_specifica.csv")

    # comunalități = partea din variabilă explicată de factorii comuni
    comm = model_fact.get_communalities()
    t_comm = pd.Series(np.append(comm, sum(comm)), index=variabile_observate + ["Total"], name="Comunalitate")

    corelograma(pd.DataFrame(t_comm), "Comunalitati", vmin=0, cmap="Blues", annot=True, vmax=1)
    corelograma(pd.DataFrame(t_psi), "Varianta specifica", vmin=0, cmap="Reds", annot=True)

    # 5) Alegere nr. factori (similar PCA): aici plot_varianta pe varianta[0]
    k = plot_varianta(varianta[0], procent_minimal=70, eticheta_x="Factor")
    nr_min_factori = min([v for v in k if v is not None])
    print("[FA] Numar factori semnificativi:", nr_min_factori)

    # 6) Încărcături factoriale (loadings)
    l = model_fact.loadings_
    etichete_factori = list(t_varianta.index)

    t_l = salvare_ndarray(
        l,
        variabile_observate,
        etichete_factori,
        nume_index="Indicatori",
        nume_fisier_output="data_out_fa/l.csv"
    )
    corelograma(t_l, "Corelograma - AFact", annot=m <= 10)

    for i in range(2, nr_min_factori + 1):
        scatterplot(
            t_l,
            x="F1",
            y="F" + str(i),
            titlu="Plot corelatii - AFact",
            etichete=t_l.index,
            corelatii=True
        )

    # 7) Scoruri pe factori
    f_scores = model_fact.transform(x)
    t_f = salvare_ndarray(
        f_scores,
        set_date.index,
        etichete_factori,
        nume_index=set_date.index.name,
        nume_fisier_output="data_out_fa/f.csv"
    )

    for i in range(2, nr_min_factori + 1):
        scatterplot(
            t_f,
            x="F1",
            y="F" + str(i),
            titlu="Plot scoruri - AFact",
            etichete=t_f.index
        )

    # 8) Hărți scoruri pe factori
    gdf = GeoDataFrame.from_file("RO_NUTS2/Ro.shp")
    camp_legatura = "sj"
    for i in range(1, nr_min_factori + 1):
        plot_harta(gdf, camp_legatura, t_f, "F" + str(i), titlu="Harta scoruri")

    print(f"[FA] Gata. Am selectat nr_min_factori = {nr_min_factori}.")
    return nr_min_factori


# =============================================================================
# 5) PUNCT DE INTRARE (alegi ce rulezi)
# =============================================================================

if __name__ == "__main__":
    """
    Alege ce vrei să rulezi:

    MODE = "pca"  -> rulează doar PCA (ACP)
    MODE = "fa"   -> rulează doar Analiza factorială
    MODE = "both" -> rulează ambele (PCA apoi FA)

    Recomandare:
    - dacă ai nevoie și de outputurile din data_out și data_out_fa, folosește "both".
    """
    MODE = "both"   # <<< schimbă aici dacă vrei doar una

    if MODE == "pca":
        run_pca()
        show()

    elif MODE == "fa":
        run_fa()
        show()

    elif MODE == "both":
        run_pca()
        run_fa()
        show()

    else:
        raise ValueError("MODE invalid. Folosește: 'pca', 'fa' sau 'both'.")
