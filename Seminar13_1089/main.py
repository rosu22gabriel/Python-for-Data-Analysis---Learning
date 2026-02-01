import pandas as pd
from scipy.cluster.hierarchy import linkage

from functii import nan_replace_df, calcul_partitie
from graphics import plot_ierarhie, show

t = pd.read_csv("data_in/MortalitateRO2019/mortalitate_ro.csv",index_col=1)
nan_replace_df(t)

variabile_observate = list(t)[1:]

x = t[variabile_observate].values
metoda_grupare = "complete"
# Calcul matrice ierarhie
h = linkage(x,metoda_grupare)
print("Matricea ierarhie:")
print(h)
plot_ierarhie(h,t.index)
show()

# Calcul si analiza partitie optimala
k_opt,color_threshold_opt,p_opt = calcul_partitie(h)
print("Numar clusteri in partitia optimala:",k_opt)
print("Distanta de sectionare in partitia optimala:",color_threshold_opt)
plot_ierarhie(h,t.index,
              color_threshold_opt,
              "Partitia optimala"
              )
t_partitii = pd.DataFrame(
    data={
        "Partitie O":p_opt
    }, index=t.index
)
# show()

# Calcul si analiza partitie din 3 clusteri
k,color_threshold_k,p_k = calcul_partitie(h,3)
plot_ierarhie(
    h,
    t.index,
    color_threshold_k,
    "Partitia din "+str(k)+" clusteri"
)
t_partitii["Partitie "+str(k)] = p_k
show()

t_partitii.to_csv("data_out/Partitii.csv")
