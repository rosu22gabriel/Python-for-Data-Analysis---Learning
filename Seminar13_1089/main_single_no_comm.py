import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
from scipy.cluster.hierarchy import linkage, dendrogram


def nan_replace_df(t: pd.DataFrame) -> None:
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c: t[c].mode()[0]}, inplace=True)


def salvare_ndarray(x: np.ndarray, nume_linii, nume_coloane, nume_fisier_output="out.csv") -> None:
    temp = pd.DataFrame(x, index=nume_linii, columns=nume_coloane)
    temp.to_csv(nume_fisier_output)


def calcul_partitie(h: np.ndarray, k: int | None = None):
    m = h.shape[0]
    n = m + 1

    if k is None:
        diferente = h[1:, 2] - h[:m - 1, 2]

        j = np.argmax(diferente) + 1

        k = n - j
    else:
        j = n - k

    color_threshold = (h[j, 2] + h[j - 1, 2]) / 2

    c = np.arange(n)

    for i in range(j):
        k1 = h[i, 0]
        k2 = h[i, 1]

        c[c == k1] = n + i
        c[c == k2] = n + i

    partitie = ["C" + str(i + 1) for i in pd.Categorical(c).codes]

    return k, color_threshold, np.array(partitie)


def plot_ierarhie(h: np.ndarray, etichete=None, color_threshold: float = 0, titlu: str = "Plot Ierarhie"):
    f = plt.figure(titlu, figsize=(10, 6))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16})

    dendrogram(h, color_threshold=color_threshold, labels=etichete, ax=ax)

    if color_threshold != 0:
        ax.axhline(color_threshold, c="r")


def show():
    plt.show()

def main():

    input_path = "data_in/MortalitateRO2019/mortalitate_ro.csv"
    t = pd.read_csv(input_path, index_col=1)

    nan_replace_df(t)

    variabile_observate = list(t)[1:]

    x = t[variabile_observate].values

    metoda_grupare = "complete"

    h = linkage(x, metoda_grupare)

    print("Matricea ierarhie (linkage matrix):")
    print(h)

    plot_ierarhie(h, t.index)
    show()

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


    t_partitii = pd.DataFrame(
        data={"Partitie O": p_opt},
        index=t.index
    )

    k = 3
    k_fixed, color_threshold_k, p_k = calcul_partitie(h, k)


    plot_ierarhie(
        h,
        t.index,
        color_threshold_k,
        "Partitia din " + str(k_fixed) + " clusteri"
    )
    t_partitii["Partitie " + str(k_fixed)] = p_k

    show()

    os.makedirs("data_out", exist_ok=True)
    output_path = "data_out/Partitii.csv"
    t_partitii.to_csv(output_path)

    print("\nAm salvat partițiile în:", output_path)
    print("\nPreview (primele 10 rânduri):")
    print(t_partitii.head(10))


if __name__ == "__main__":
    main()
