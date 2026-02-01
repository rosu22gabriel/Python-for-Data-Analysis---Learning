"""
Seminar 13 – Clusterizare ierarhică (totul într-un singur fișier)

Ce face acest script, pe scurt:
1) Citește dataset-ul de mortalitate (CSV).
2) Înlocuiește valorile lipsă (NaN):
   - numeric -> cu media coloanei
   - nenumeric -> cu moda (cel mai frecvent)
3) Construiește ierarhia de clusterizare (linkage) folosind metoda "complete".
4) Desenează dendrograma (plot ierarhie).
5) Determină o partiție "optimă" cu o regulă tip Elbow aplicată pe distanțele de aglomerare.
6) Mai calculează și o partiție pentru k=3 clustere.
7) Salvează partițiile într-un fișier CSV (data_out/Partitii.csv)

Dependențe:
- pandas
- numpy
- matplotlib
- scipy

Instalare (dacă ai nevoie):
pip install pandas numpy matplotlib scipy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
from scipy.cluster.hierarchy import linkage, dendrogram


# ============================================================
# 1) FUNCȚII UTILITARE (înainte erau în functii.py)
# ============================================================

def nan_replace_df(t: pd.DataFrame) -> None:
    """
    Înlocuiește valorile lipsă (NaN) dintr-un DataFrame, "in place" (adică modifică direct t).

    Logică:
    - Parcurgem fiecare coloană.
    - Dacă există NaN în coloană:
        * dacă este numerică -> completăm NaN cu media acelei coloane
        * dacă nu e numerică -> completăm NaN cu moda (valoarea cea mai frecventă)

    De ce e util:
    - Algoritmii de clusterizare (și multe calcule numerice) NU funcționează cu NaN.
    - Completarea cu media/moda e o metodă simplă și standard pentru "imputare".
    """
    for c in t.columns:
        # t[c].isna() produce un vector True/False pe fiecare rând; any(...) verifică dacă există măcar un True.
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                # Pentru numeric: media coloanei (ignoră NaN automat).
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                # Pentru nenumeric: moda (mode() poate întoarce mai multe valori; o alegem pe prima).
                t.fillna({c: t[c].mode()[0]}, inplace=True)


def salvare_ndarray(x: np.ndarray, nume_linii, nume_coloane, nume_fisier_output="out.csv") -> None:
    """
    Salvează un numpy array ca tabel CSV, folosind pandas.

    Parametri:
    - x: matricea (ndarray) de salvat
    - nume_linii: index (etichetele rândurilor)
    - nume_coloane: numele coloanelor
    - nume_fisier_output: unde salvăm CSV

    Notă:
    - În proiectul tău original funcția există, chiar dacă aici nu e neapărat folosită în main.
    """
    temp = pd.DataFrame(x, index=nume_linii, columns=nume_coloane)
    temp.to_csv(nume_fisier_output)


def calcul_partitie(h: np.ndarray, k: int | None = None):
    """
    Calculează o partiție (împărțirea observațiilor în clustere) pornind de la matricea de linkage (h).

    Ce este h (linkage matrix):
    - Are forma (m, 4), unde m = n-1 (n = număr observații).
    - Fiecare rând i reprezintă o "lipire" (merge) între două clustere:
        h[i, 0] = id cluster 1
        h[i, 1] = id cluster 2
        h[i, 2] = distanța la care s-a făcut lipirea
        h[i, 3] = numărul de observații din clusterul rezultat (nu-l folosim aici)

    Scopul funcției:
    - Dacă k este None -> găsește o partiție "optimă" automat (folosind o idee de tip Elbow).
    - Dacă k este dat -> returnează partiția cu EXACT k clustere.

    Returnează:
    - k (număr clustere)
    - color_threshold (înălțimea la care "tăiem" dendrograma)
    - partitie (vector de etichete "C1", "C2", ... pentru fiecare observație)

    ------------------------------------------------------------
    Cum decide "partiția optimă" când k=None?
    1) h[:,2] sunt distanțele de aglomerare (cât de "departe" trebuie să mergem ca să lipim clustere).
    2) Ne uităm la diferențe succesive între distanțe: h[i+1,2] - h[i,2]
       -> unde apare "saltul" cel mai mare în distanță, înseamnă că de acolo în sus se lipesc clustere
          mult mai diferite, deci e un prag bun de tăiere (heuristică Elbow).
    3) Alegem indexul j unde saltul e maxim, iar k = n - j.

    ------------------------------------------------------------
    Cum construim efectiv etichetele de cluster?
    - Pornim de la partiția singleton: fiecare observație e în clusterul ei.
    - Simulăm primele j lipiri (merge-uri):
        la fiecare merge i, înlocuim în vectorul c id-urile celor două clustere unite cu id-ul nou creat.
      În SciPy, clusterele nou create au id-uri n+i.
    - La final, transformăm id-urile în coduri 0..(nr_clustere-1) și punem etichete C1, C2, ...
    """
    m = h.shape[0]      # m = n - 1
    n = m + 1           # n = număr observații

    if k is None:
        # -------------------------------
        # 1) Partiția "optimă" (Elbow)
        # -------------------------------
        # h[1:,2] - h[:m-1,2] => diferențe dintre distanțe consecutive.
        diferente = h[1:, 2] - h[:m - 1, 2]

        # np.argmax => poziția celui mai mare salt.
        # +1 deoarece diferente are lungimea m-1 (compară rândurile 0-1, 1-2 etc)
        j = np.argmax(diferente) + 1

        # Dacă am făcut j merge-uri, au rămas n-j clustere.
        k = n - j
    else:
        # -------------------------------
        # 2) Partiția cu k clustere
        # -------------------------------
        # Dacă vrem k clustere, trebuie să facem exact j = n - k lipiri.
        j = n - k

    # -----------------------------------
    # Alegem înălțimea (pragul) la care "tăiem" dendrograma.
    # Intuiție:
    # - Ca să obții k clustere, tai între merge-ul j-1 și merge-ul j.
    # - Dacă alegi un prag între distanțele h[j-1,2] și h[j,2], atunci primele j merge-uri se execută,
    #   iar următoarele NU (deci rămâi cu n-j = k clustere).
    # Pragul ales este media celor două distanțe vecine:
    color_threshold = (h[j, 2] + h[j - 1, 2]) / 2

    # -----------------------------------
    # Construim partiția efectivă
    # c pornește ca [0,1,2,...,n-1] (fiecare observație e cluster separat)
    c = np.arange(n)

    # Simulăm primele j merge-uri din h
    for i in range(j):
        k1 = h[i, 0]
        k2 = h[i, 1]

        # Înlocuim toate aparițiile clusterelor k1 și k2 cu id-ul nou (n+i)
        # Atenție: k1/k2 pot fi fie id-uri de observații (0..n-1),
        #          fie id-uri de clustere create anterior (>= n).
        c[c == k1] = n + i
        c[c == k2] = n + i

    # c conține id-uri de clustere (nu neapărat 0..k-1).
    # pd.Categorical(c).codes transformă valorile distincte în coduri consecutive 0..k-1.
    # Apoi facem etichete "C1", "C2", ...
    partitie = ["C" + str(i + 1) for i in pd.Categorical(c).codes]

    return k, color_threshold, np.array(partitie)


# ============================================================
# 2) FUNCȚII DE GRAFICE (înainte erau în graphics.py)
# ============================================================

def plot_ierarhie(h: np.ndarray, etichete=None, color_threshold: float = 0, titlu: str = "Plot Ierarhie"):
    """
    Desenează dendrograma (arborele ierarhic).

    Parametri:
    - h: linkage matrix (de la scipy.cluster.hierarchy.linkage)
    - etichete: etichete pentru frunze (de obicei numele observațiilor)
    - color_threshold:
        * dacă e 0 -> SciPy colorează standard (fără prag special)
        * dacă e >0 -> ramurile sub prag sunt colorate diferit; practic vezi "tăietura" pentru partiție
    - titlu: titlul figurii

    Ce reprezintă dendrograma:
    - Pe axa verticală (y) ai distanța la care se lipesc clusterele.
    - Dacă trasezi o linie orizontală (prag) și "tai" arborele:
        numărul de componente/ramuri separate sub linie = numărul de clustere.
    """
    f = plt.figure(titlu, figsize=(10, 6))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16})

    dendrogram(h, color_threshold=color_threshold, labels=etichete, ax=ax)

    # Dacă avem prag de tăiere, trasăm și o linie roșie ca să se vadă clar.
    if color_threshold != 0:
        ax.axhline(color_threshold, c="r")


def show():
    """Wrapper simplu peste plt.show() (păstrat ca în proiectul original)."""
    plt.show()


# ============================================================
# 3) MAIN – fluxul efectiv al aplicației
# ============================================================

def main():
    # ------------------------------------------------------------
    # (A) Citirea datelor
    # ------------------------------------------------------------
    # Observație importantă: în proiectul tău original ai index_col=1,
    # adică a doua coloană din fișier devine index (etichetele rândurilor).
    #
    # Asta e util când ai un identificator (ex: județ / regiune / categorie) pe acea coloană.
    input_path = "data_in/MortalitateRO2019/mortalitate_ro.csv"
    t = pd.read_csv(input_path, index_col=1)

    # ------------------------------------------------------------
    # (B) Tratarea valorilor lipsă
    # ------------------------------------------------------------
    nan_replace_df(t)

    # ------------------------------------------------------------
    # (C) Alegerea variabilelor folosite la clusterizare
    # ------------------------------------------------------------
    # În proiectul original: variabile_observate = list(t)[1:]
    # Asta înseamnă:
    # - luăm lista coloanelor
    # - ignorăm prima coloană (poate e un alt id / metadată)
    #
    # ATENȚIE: asta e o convenție bazată pe structura fișierului tău.
    # Dacă prima coloană e deja numerică și relevantă, atunci [1:] ar exclude-o.
    variabile_observate = list(t)[1:]

    # Extragem matricea X (doar valorile numerice ale variabilelor selectate).
    # .values => numpy array de dimensiune (n_observații, n_variabile)
    x = t[variabile_observate].values

    # ------------------------------------------------------------
    # (D) Clusterizare ierarhică: construim matricea linkage
    # ------------------------------------------------------------
    metoda_grupare = "complete"

    # linkage(x, metoda_grupare) produce matricea h (n-1 rânduri, 4 coloane)
    # "complete" (complete linkage) = distanța dintre două clustere este
    # distanța MAXIMĂ dintre oricare două puncte (unul din primul cluster, unul din al doilea).
    #
    # Intuiție:
    # - Complete linkage tinde să producă clustere mai compacte și evită clustere foarte alungite.
    h = linkage(x, metoda_grupare)

    print("Matricea ierarhie (linkage matrix):")
    print(h)

    # Plot dendrogramă fără prag special (doar ierarhia brută)
    plot_ierarhie(h, t.index)
    show()

    # ------------------------------------------------------------
    # (E) Partiția "optimă" (k automat) – heuristică Elbow
    # ------------------------------------------------------------
    k_opt, color_threshold_opt, p_opt = calcul_partitie(h)

    print("\n--- Partitia optimala (heuristica Elbow) ---")
    print("Numar clusteri in partitia optimala:", k_opt)
    print("Distanta de sectionare in partitia optimala (prag):", color_threshold_opt)

    plot_ierarhie(
        h,
        t.index,
        color_threshold_opt,
        "Partitia optimala"
    )

    # Construim un DataFrame cu partițiile, ca să le salvăm ușor în CSV.
    t_partitii = pd.DataFrame(
        data={"Partitie O": p_opt},
        index=t.index
    )

    # ------------------------------------------------------------
    # (F) Partiția cu k = 3 clustere (caz impus)
    # ------------------------------------------------------------
    k = 3
    k_fixed, color_threshold_k, p_k = calcul_partitie(h, k)

    # k_fixed ar trebui să fie 3 (doar ca verificare)
    plot_ierarhie(
        h,
        t.index,
        color_threshold_k,
        "Partitia din " + str(k_fixed) + " clusteri"
    )

    # Adăugăm și coloana pentru această partiție
    t_partitii["Partitie " + str(k_fixed)] = p_k

    # Afișăm graficele rămase (partiția optimă + partiția k=3)
    show()

    # ------------------------------------------------------------
    # (G) Salvare rezultate
    # ------------------------------------------------------------
    os.makedirs("data_out", exist_ok=True)
    output_path = "data_out/Partitii.csv"
    t_partitii.to_csv(output_path)

    print("\nAm salvat partițiile în:", output_path)
    print("\nPreview (primele 10 rânduri):")
    print(t_partitii.head(10))


if __name__ == "__main__":
    main()
