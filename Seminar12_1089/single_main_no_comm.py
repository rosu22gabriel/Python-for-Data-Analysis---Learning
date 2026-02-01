import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seaborn import kdeplot, scatterplot
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import f

def calcul_metrici(y_true, y_pred, clase):
    cm = confusion_matrix(y_true, y_pred, labels=clase)

    t_cm = pd.DataFrame(cm, index=clase, columns=clase)

    t_cm["Acuratete"] = np.diag(cm) * 100 / np.sum(cm, axis=1)

    acuratete_g = np.sum(np.diag(cm)) * 100 / len(y_true)

    acuratete_m = t_cm["Acuratete"].mean()

    i_ck = cohen_kappa_score(y_true, y_pred, labels=clase)

    acuratete = pd.Series(
        [acuratete_g, acuratete_m, i_ck],
        index=["Acuratete globala", "Acuratete medie", "Index CK"],
        name="Indicatori acuratete"
    )

    return t_cm, acuratete


def salvare_ndarray(x: np.ndarray, nume_linii, nume_coloane, nume_fisier_output="out.csv"):
    temp = pd.DataFrame(x, index=nume_linii, columns=nume_coloane)
    temp.to_csv(nume_fisier_output)
    return temp


def f_distributii(t: pd.DataFrame, variabila, y, clase, titlu="Distributii"):
    fig = plt.figure(f"{titlu} - {variabila}", figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"{titlu} - {variabila}", fontdict={"fontsize": 16})

    kdeplot(t, x=variabila, hue=y, hue_order=clase, ax=ax, fill=True)


def f_scatter(t: pd.DataFrame, tg: pd.DataFrame, y, clase,
              varx="Z1", vary="Z2", titlu="Plot instante in axe discriminante"):
    fig = plt.figure(f"{titlu} - {varx} {vary}", figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16})

    scatterplot(t, x=varx, y=vary, hue=y, hue_order=clase, ax=ax)

    scatterplot(tg, x=varx, y=vary, hue=clase, hue_order=clase, legend=False,
                ax=ax, marker="s", s=100)


def show():
    plt.show()


def main():
    t = pd.read_csv("data_in/Hernia/hernia.csv", index_col=0)

    variabile = list(t)

    tinta = variabile[-1]

    predictori = variabile[:-1]

    os.makedirs("data_out", exist_ok=True)

    t_train, t_test, y_train, y_test = train_test_split(
        t[predictori], t[tinta], test_size=0.3
    )

    model_lda = LinearDiscriminantAnalysis()
    model_lda.fit(t_train, y_train)

    clase = model_lda.classes_

    x = t_train[predictori].values
    x_ = np.mean(x, axis=0)

    g = model_lda.means_
    n = len(t_train)
    ponderi = model_lda.priors_
    q = len(ponderi)

    dg = np.diag(ponderi) * n

    sst = (x - x_).T @ (x - x_)

    ssb = (g - x_).T @ dg @ (g - x_)

    ssw = sst - ssb

    f_predictori = (np.diag(ssb) / (q - 1)) / (np.diag(ssw) / (n - q))

    pvalues = 1 - f.cdf(f_predictori, q - 1, n - q)

    t_predictori = pd.DataFrame(
        data={"Putere discriminare": f_predictori, "PValues": pvalues},
        index=predictori
    )
    t_predictori.to_csv("data_out/Predictori.csv")

    for predictor in predictori:
        f_distributii(t_train, predictor, y_train, clase)

    predictie_test_lda = model_lda.predict(t_test)

    t_predictii_test = pd.DataFrame(
        data={tinta: y_test, "Predictie LDA": predictie_test_lda},
        index=t_test.index
    )
    t_predictii_test.to_csv("data_out/Predictii_test.csv")

    t_cm, acuratete = calcul_metrici(y_test, predictie_test_lda, clase)
    t_cm.to_csv("data_out/CM_lda.csv")
    acuratete.to_csv("data_out/Acuratete_lda.csv")


    z = model_lda.transform(t_train)
    etichete_z = [f"Z{i+1}" for i in range(q - 1)]

    t_z = salvare_ndarray(z, t_train.index, etichete_z, "data_out/z.csv")

    t_gz = t_z.groupby(by=y_train).mean()
    gz = t_gz.values


    if (q - 1) >= 2:
        f_scatter(t_z, t_gz, y_train, clase, varx="Z1", vary="Z2")

    z_ = np.mean(z, axis=0)

    sst_z = (z - z_).T @ (z - z_)
    ssb_z = (gz - z_).T @ dg @ (gz - z_)
    ssw_z = sst_z - ssb_z

    f_z = (np.diag(ssb_z) / (q - 1)) / (np.diag(ssw_z) / (n - q))
    pvalues_z = 1 - f.cdf(f_z, q - 1, n - q)

    print("F pe axele Z:", f_z)
    print("p-values pe axele Z:", pvalues_z)

    show()


if __name__ == "__main__":
    main()
