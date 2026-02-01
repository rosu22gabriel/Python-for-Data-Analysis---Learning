import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,cohen_kappa_score

def calcul_metrici(y,predictie,clase):
    cm = confusion_matrix(y,predictie,labels=clase)
    # print(cm)
    t_cm = pd.DataFrame(cm,clase,clase)
    t_cm["Acuratete"] = np.diag(cm)*100/np.sum(cm,axis=1)
    acuratete_g = sum(np.diag(cm))*100/len(y)
    acuratete_m = t_cm["Acuratete"].mean()
    i_ck = cohen_kappa_score(y,predictie,labels=clase)
    acuratete = pd.Series(
        [acuratete_g,acuratete_m,i_ck],["Acuratete globala","Acuratete medie","Index CK"],
        name="Indicatori acuratete"
    )
    return t_cm,acuratete

def salvare_ndarray(x:np.ndarray,nume_linii,nume_coloane,nume_fisier_output="out.csv"):
    temp = pd.DataFrame(x,nume_linii,nume_coloane)
    temp.to_csv(nume_fisier_output)
    return temp
