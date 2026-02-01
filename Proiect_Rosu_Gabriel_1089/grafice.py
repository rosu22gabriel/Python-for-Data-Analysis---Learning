import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def plot_ierarhie(h: np.ndarray, etichete=None, color_threshold=0, titlu="Plot ierarhie"):
    f = plt.figure(titlu, figsize=(10, 6))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16})
    dendrogram(h, color_threshold=color_threshold, labels=etichete, ax=ax)
    if color_threshold != 0:
        ax.axhline(color_threshold, c="r")


def plot_silhouette(s_vals: np.ndarray, partitii: np.ndarray, titlu="Grafic Silhouette"):
    et = np.array(partitii)
    clustere = np.unique(et)

    y_lower = 10
    f = plt.figure(titlu, figsize=(10, 6))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16})
    ax.set_xlabel("Valoare Silhouette")
    ax.set_ylabel("Cluster")

    # optional dar recomandat: scara clasica silhouette
    ax.set_xlim([-1, 1])

    for c in clustere:
        vals_c = s_vals[et == c]
        vals_c = np.sort(vals_c)[::-1]
        size_c = vals_c.shape[0]
        y_upper = y_lower + size_c

        ax.barh(range(y_lower, y_upper), vals_c)
        ax.text(-0.05, (y_lower + y_upper) / 2, str(c))
        y_lower = y_upper + 10

    ax.axvline(np.mean(s_vals), linestyle="--")


def plot_scatter_axe_principale(
    scoruri_2d: np.ndarray,
    partitii: np.ndarray,
    etichete=None,
    titlu="Plot instante pe primele doua axe principale"
):
    et = np.array(partitii)

    f = plt.figure(titlu, figsize=(10, 6))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16})
    ax.set_xlabel("Axa principala 1")
    ax.set_ylabel("Axa principala 2")

    clustere = np.unique(et)
    for c in clustere:
        idx = np.where(et == c)[0]
        ax.scatter(scoruri_2d[idx, 0], scoruri_2d[idx, 1], label=str(c))

    if etichete is not None:
        for i, txt in enumerate(etichete):
            ax.text(scoruri_2d[i, 0], scoruri_2d[i, 1], str(txt), fontsize=9)

    ax.legend()


def histograme_pe_clusteri(
    df,
    coloane: list,
    col_cluster="Cluster",
    titlu_prefix="Histograma",
    salveaza=False,
    folder_out="data_out"
):
    clustere = sorted(df[col_cluster].unique())

    for col in coloane:
        f = plt.figure(f"{titlu_prefix} - {col}", figsize=(10, 6))
        ax = f.add_subplot(1, 1, 1)
        ax.set_title(f"{titlu_prefix} - {col}", fontdict={"fontsize": 16})
        ax.set_xlabel(col)
        ax.set_ylabel("Frecventa")

        for c in clustere:
            valori = df[df[col_cluster] == c][col].dropna().values
            # c este deja C1/C2/... deci NU mai punem inca un C
            ax.hist(valori, bins=6, alpha=0.6, label=str(c))

        ax.legend()

        if salveaza:
            os.makedirs(folder_out, exist_ok=True)
            nume = (
                str(col)
                .replace("/", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("%", "proc")
                .replace(" ", "_")
            )
            f.savefig(f"{folder_out}/hist_{nume}.png", dpi=200, bbox_inches="tight")


def show():
    plt.show()
