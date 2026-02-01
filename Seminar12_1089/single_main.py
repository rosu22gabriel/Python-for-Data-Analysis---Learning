"""
Seminar 12 - Analiza discriminanta liniara (LDA)

Scopul seminarului:
- Avem un set de date (hernia.csv) cu mai multi predictori (variabile explicative)
 si o variabila tinta (clasa / diagnosticul).
- Vrem sa construim un model LDA care:
1) Invata din date (train)
2) Prezice classa pe date noi (test)
3) Evalueaza cat de bine e prezis (matrice de confuzie, acuratete, Cohen Kappa)
4) Analizeaza "axele discriminante" (scoruri Z1, Z2, ...), adica proiectiile datelor
    intr-un spatiu unde clasele sunt cat mai separate

Ce fisiere genereaza in data_out/:
- Predictori.csv     -> puterea de discriminare + p-value pentru fiecare predictor
- Predictii_test.csv -> real vs predictie pe setul de test
- CM_lda.csv         -> matrice de confuzie + acuratete pe fiecare clasa
- Acuratete_lda.csv  -> acuratete globala, medie si Cohen Kappa
- z.csv              -> scorurile discriminante ale observatiilor din train (Z1, Z2, ...)

Dependente:
 - numpy, pandas, matplotlib
 - seaborn (pentru grafice)
 - scikit-learn (train_test_split, LDA, confusion_matrix, cohen_kappa_score)
 - scipt (distributia F pentru p-value)

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seaborn import kdeplot, scatterplot
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.stats import f

# ------------------------
# 1) Functii utilitare
# ------------------------

def calcul_metrici(y_true, y_pred, clase):
    """
    Calculeaza metrici de clasificare pentru un model (aici LDA)

    Parametri:
    - y_true: etichetele reale (vector/Series)
    - y_pred: etichetele prezise (vector/Series)
    - clase: lista / array cu clasele posibile, in ordinea dorita (important pt CM)

    Intoarce:
    - t_cm: DataFrame cu matricea de confuzie + acuratete pe fiecare clasa
    - acuratete: Series cu:
        * Acuratete globala (overall accuracy)
        * Acuratete medie (media acuratetilor pe clase)
        * Index CK (Cohen Kappa) - cat de mult depasim "ghicitul" aleator

    -----------------------------------------------------------------
    Ce este matricea de confuzie?
    - randuri = clasa reala
    - coloane = clasa prezisa
    Exemplu: cm[i, j] = cate observatii din clasa i au fost prezise ca j.

    Acuratete pe clasa:
    - pe fiecare rand (clasa reala), ce procent a fost prezis corect:
        diag(cm) / suma_randului

    Acuratete globala:
    - total corecte / total observatii

    Cohen Kappa:
    - o metrica ce tine cont de acordul "din intamplare"
    - 1 = perfect, 0 = ca aleator, <0 = mai rau ca aleator (rar).
    """
    cm = confusion_matrix(y_true, y_pred, labels=clase)

    # Convertim matricea numpy intr-un tabel usor de citit.
    t_cm = pd.DataFrame(cm, index=clase, columns=clase)

    # Acuratete pe fiecare clasa (pe rand):
    # diag(cm) = elementele corecte pentru fiecare clasa
    # sum(cm, axis=1) = total observatii reale din fiecare clasa
    t_cm["Acuratete"] = np.diag(cm) * 100 / np.sum(cm, axis=1)

    # Acuratete globala: total corecte / total
    acuratete_g = np.sum(np.diag(cm)) * 100 / len(y_true)

    # Acuratete medie: media acuratetilor pe clase
    acuratete_m = t_cm["Acuratete"].mean()

    # Cohen Kappa
    i_ck = cohen_kappa_score(y_true, y_pred, labels=clase)

    # Le punem intr-un Series ca sa fie usor de salvat/afisat
    acuratete = pd.Series(
        [acuratete_g, acuratete_m, i_ck],
        index=["Acuratete globala", "Acuratete medie", "Index CK"],
        name="Indicatori acuratete"
    )

    return t_cm, acuratete

def salvare_ndarray(x: np.ndarray, nume_linii, nume_coloane, nume_fisier_output="out.csv"):
    """
    Salveaza un numpy array ca tabel CSV, folosind pandas

    Parametri:
    - x: array/matrice numpy
    - nume_linii: indexul (etichetele randurilor) pentru DataFrame
    - nume_coloane: numele coloanelor
    - nume_fisier_output: fisierul CSV unde salvam


    Returneaza: - DataFrame-ul creat (util daca vrei sa mai lucrezi cu el dupa salvare).
    """
    temp = pd.DataFrame(x, index=nume_linii, columns=nume_coloane)
    temp.to_csv(nume_fisier_output)
    return temp

# --------------------------------------------
# Functii de grafice
# --------------------------------------------

def f_distributii(t: pd.DataFrame, variabila, y, clase, titlu="Distributii"):
    """
    Deseneaza distributiile (densitati KDE) ale unei variabile pe clase.

    De ce ajuta?
    - Daca distributiile pe clase sunt bine separate, predictorul are sanse sa fie bun
    - Daca se suprapun foarte mult, predictorul singur nu discrimineaza bine.

    Parametri:
    - t: DataFrame cu datele (de obicei setul de train)
    - variabila: numele predictorului (coloana)
    - y: etichetele (clasa) pentru fiecare observatie (Series/array)
    - clase: ordinea claselor (pentru legenda consistenta)
    - titlu: titlul general al figurii
    """
    fig = plt.figure(f"{titlu} - {variabila}", figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"{titlu} - {variabila}", fontdict={"fontsize": 16})

    # kdeplot cu hue = y => o densitate per clasa
    kdeplot(t, x=variabila, hue=y, hue_order=clase, ax=ax, fill=True)


def f_scatter(t: pd.DataFrame, tg: pd.DataFrame, y, clase,
              varx="Z1", vary="Z2", titlu="Plot instante in axe discriminante"):
    """
    Scatter plot pe axele discriminante (Z1 vs Z2), colorat pe clase.
    In plus, ploteaza si centroizii (mediile) claselor in spatiul Z.

    Parametri:
    - t: tabelul cu scoruri Z pentru fiecare observatie (ex: t_z)
    - tg: tabelul cu mediile pe clase (ex: t_gz)
    - y: etichetele de clasa ale observatiilor (pentru colorare)
    - clase: ordinea claselor
    - varx, vary: ce axe desenam (Z1, Z2 etc.)
    - titlu: titlul figurii

    Interpretare:
    - Daca punctele din clase diferite sunt bine separate in Z1/Z2,
    LDA a gasit o protectie buna pentru discriminare.
    - Patratele (marker "s") sunt centroizii claselor.
    """

    fig = plt.figure(f"{titlu} - {varx} {vary}", figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16})

    scatterplot(t, x=varx, y=vary, hue=y, hue_order=clase, ax=ax)

def show():
    """Wrapper simplu ca sa pastram aceeasi structura ca in proiect"""
    plt.show()

# ----------------------------------------
# 3) Main - fluxul complet al seminarului
# ----------------------------------------

def main():
    # --------------------------------------------
    # A) Citirea datelor
    # --------------------------------------------
    # index_col=0 => prima coloana din CSV devine index (de obicei un ID)
    t = pd.read_csv("data_in/Hernia/hernia.csv", index_col=0)

    # variabile = lista tuturor coloanelor din dataset
    variabile = list(t)

    # conventie: utlima coloana este tinta (clasa)
    tinta = variabile[-1]

    # restul coloanelor sunt predictori
    predictori = variabile[:-1]

    # Asiguram folderul de output
    os.makedirs("data_out", exist_ok=True)

    # -------------------------------------------
    # B) Splitare train/test
    # -------------------------------------------
    # Impartim datele in:
    # - train: modelul invata parametrii (mediile claselor, matricea de covarianta etc.)
    # - test: verificam cat de bine generalizeaza pe date nevazute
    #
    # test_size=0.3 => 30% test, 70% train
    #
    # Nota: In codul original nu exista random_state => rezultatele se pot schimba la fiecare rulare
    t_train, t_test, y_train, y_test = train_test_split(
        t[predictori], t[tinta], test_size=0.3
    )

    # -------------------------------------------
    # C) Antrenarea modelului LDA
    # -------------------------------------------
    model_lda = LinearDiscriminantAnalysis()
    model_lda.fit(t_train, y_train)

    # Clasele identificate de model (ordinea lor conteaza pt matricea de confuzie)
    clase = model_lda.classes_

    # ------------------------------------------
    # D) Evaluarea "puterii de discriminare" pentru fiecare predictor
    # ------------------------------------------
    # In codul original se calculeaza manual niste marimi tip scatter matrices:
    # - SST: Imprastirea totala
    # - SSB: Imprastierea intre clase
    # - SSW: Imprastierea in interiorul claselor
    #
    # Apoi se face un "F statistic" pe fiecare variabila:
    # F_j = (SSB_j/(q - 1)) / (SSW_j/(n-q))
    #
    # unde:
    # - q = numarul de clase
    # - n = numarul de observatii din train
    #
    # Intuitie:
    # - daca o variabila are SSB mare (clasele sunt "departe" pe acea variabila)
    #  si SSW mic (in interiorul fiecarei clase valorile sunt "stranse"),
    # atunci variabila separa bine clasele => F mare.

    x = t_train[predictori].values
    x_ = np.mean(x, axis=0)         # media globala pe fiecare predictor


    g = model_lda.means_            # mediile pe clase (cate un rand per clasa)
    n = len(t_train)                # numar de observatii in train
    ponderi = model_lda.priors_     # probabilitatile claselor estimate (priors)
    q = len(ponderi)                # numar clase

    # dg = diag(priors)*n => efecctiv "marimea" ponderata a fiecarei clase
    dg = np.diag(ponderi) * n

    # SST (scatter total) = sum_i (x_i - x_mean)(x_i - x_mean)^T
    sst = (x - x_).T @ (x - x_)

    # SSB (scatter intre clase) = sum_g n_g (mean_g - x_mean)(mean_g - x_mean) ^ T
    ssb = (g - x_).T @ dg @ (g - x_)

    # SSW = SST - SSB (scatter in interiorul claselor)
    ssw = sst - ssb

    # F statistic pe variabila (folosim doar diagonalele: variabila cu variabila)
    f_predictori = (np.diag(ssb) / (q - 1)) / (np.diag(ssw) / (n - q))

    # p_value din distributia F (1 - CDF)
    pvalues = 1 - f.cdf(f_predictori, q - 1, n - q)

    # Salvam tabelul cu evaluarea predictorilor
    t_predictori = pd.DataFrame(
        data={"Putere discriminare": f_predictori, "PValues": pvalues},
        index=predictori
    )
    t_predictori.to_csv("data_out/Predictori.csv")

    # --------------------------------------------
    # E) Grafice de distributie pe fiecare predictor (train)
    # -------------------------------------------

    # Pentru fiecare predictor, desenam densitatile pe clase.
    # Asta te ajuta sa "vezi" separarea dintre clase.
    for predictor in predictori:
        f_distributii(t_train, predictor, y_train, clase)

    # --------------------------------------------
    # F) Testare model pe setul de test
    # --------------------------------------------
    predictie_test_lda = model_lda.predict(t_test)

    # Salvam un tabel cu real vs predictie
    t_predictii_test = pd.DataFrame(
        data={tinta: y_test, "Predictie LDA": predictie_test_lda},
        index=t_test.index
    )
    t_predictii_test.to_csv("data_out/Predictii_test.csv")

    # --------------------------------------------
    # G) Metrici de performanta (matrice confuzie + acuratete)
    # --------------------------------------------
    t_cm, acuratete = calcul_metrici(y_test, predictie_test_lda, clase)
    t_cm.to_csv("data_out/CM_lda.csv")
    acuratete.to_csv("data_out/Acuratete_lda.csv")

    # --------------------------------------------
    # (H) Analiza scorurilor discriminante (Z)
    # --------------------------------------------
    # LDA poate proiecta datele intr-un spatiu de dimensiune (q-1).
    # unde q = numar clase.
    # Exemple:
    # - daca ai 2 clase => ai 1 axa discriminanta (Z1)
    # - daca ai 3 clase => ai 2 axe (Z1, Z2)
    # - daca ai 4 clase => ai 3 axe etc.
    z = model_lda.transform(t_train)    # transformarea pentru observatiile din train
    etichete_z = [f"Z{i+1}" for i in range(q - 1)]

    # Salvam scorurile Z intr-un CSV
    t_z = salvare_ndarray(z, t_train.index, etichete_z, "data_out/z.csv")

    # Centroizii claselor in spatiul Z (media scorurilor Z pe fiecare clasa)
    t_gz = t_z.groupby(by=y_train).mean()
    gz = t_gz.values

    # Scatter plot (Z1 vs Z2) - doar daca exista cel putin 2 axe (q-1 >= 2)
    # In codul original se apeleaza direct; daca ai doar 2 clase (q-1=1),
    # Z2 nu exista. Atunci ar trebui sa comentezi f_scatter sau sa faci un plot 1D.
    if (q - 1) >= 2:
        f_scatter(t_z, t_gz, y_train, clase, varx="Z1", vary="Z2")

    # ------------------------------------------------------
    # I) Aceeasi idee de "putere discriminare", dar in spatiul Z
    # ------------------------------------------------------
    # Practic repetam calculele SST/SSB/SSW pe scorurile Z,
    # ca sa vedem cat de bine separa efectiv axele discriminante.
    z_ = np.mean(z, axis=0)

    sst_z = (z - z_).T @ (z - z_)
    ssb_z = (gz - z_).T @ dg @ (gz - z_)
    ssw_z = sst_z - ssb_z

    f_z = (np.diag(ssb_z) / (q - 1)) / (np.diag(ssw_z) / (n - q))
    pvalues_z = 1 - f.cdf(f_z, q-1, n-q)

    print("F pe axele Z:", f_z)
    print("p_values pe axele Z:", pvalues_z)

    # Afisam toate figurile
    show()


if __name__ == "__main__":
    main()























