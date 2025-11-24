import sys

import numpy as np
import pandas as pd

from functii import nan_replace_df, acp, tabelare_varianta, salvare_ndarray
from grafice import plot_varianta, show, corelograma, scatterplot
from geopandas import GeoDataFrame

np.set_printoptions(3,sys.maxsize,suppress=True)
pd.set_option("display.max_columns",None)

set_date = pd.read_csv("data_in/Teritorial2022/Teritorial_2022.csv",index_col=0)
nan_replace_df(set_date)

variabile_observate = list(set_date)[3:]
x = set_date[variabile_observate].values

# print(x)
x_,r_v,alpha,a = acp(x)
t_r = salvare_ndarray(r_v,
                      variabile_observate,
                      variabile_observate,
                      "Indicatori",
                      "data_out/R.csv"
                      )
corelograma(t_r,annot=len(variabile_observate)<10)

# Analiza variantei componentelor
t_varianta = tabelare_varianta(alpha)
t_varianta.round(3).to_csv("data_out/Varianta.csv")
k = plot_varianta(alpha)
nr_comp = min([v for v in k if v is not None])


# Analiza corelatiilor dintre variabilele observare si componente
c = x_@a
s = c/np.sqrt(alpha)
etichete_componente = list(t_varianta.index)
t_c = salvare_ndarray(c,
                      set_date.index,
                      etichete_componente,
                      set_date.index.name,
                      "data_out/C.csv"
                      )

t_s = salvare_ndarray(s,
                      set_date.index,
                      etichete_componente,
                      set_date.index.name,
                      "data_out/S.csv"
                      )
n,m = x.shape
r_XC = np.corrcoef(x_,c,rowvar=False)[:m,m:]
t_r_XC = salvare_ndarray(
    r_XC,variabile_observate,etichete_componente,
    "Indicatori","data_out/R_XC.csv"
)
corelograma(t_r_XC,"Corelatii variabile-componente",annot=m<10)

for i in range(2, nr_comp+1):
    scatterplot(t_r_XC
                ,titlu="Corelatii factoriale"
                ,y="C" + str(i)
                ,etichete=variabile_observate
                ,corelatii=True)



# Analiza scorurilor
# Stabilim axa a 2-a
for i in range(2, nr_comp+1):
    # X este C1(Componenta 1)
    # Y va fi pe rand C2, C3...
    scatterplot(t_c,titlu="Plot componente", y="C" + str(i), etichete=set_date.index)
    scatterplot(t_s, etichete=set_date.index, y="C" + str(i))


# Calcul matrici
# Calcul cosinusuri -> scot importanta pentru o componenta in raport cu alta
# Daca e mai aproape de o axa fata de cealalta si in ce masura
c2 = c * c
# Transpunem c2 cu .T pentru a efectua calculul pe linii
cosin = (c2.T/np.sum(c2, axis=1)).T
t_cosin = salvare_ndarray(
    cosin,
    set_date.index,
    etichete_componente,
    set_date.index.name,
    "data_out/Cosin.csv"
)

if n < 50:
    corelograma(
        t_cosin,
        "Cosinusuri",
        "0",
        "Reds",
        n < 15
    )

# Contributii
contrib = c2/np.sum(c2, axis=0)

t_contrib = salvare_ndarray(
    contrib,
    set_date.index,
    etichete_componente,
    set_date.index.name,
    "data_out/Contrib.csv"
)
if n < 50:
    corelograma(
        t_contrib,
        "Contributii",
        0,
        "Reds",
        n < 15,
        100
    )

# Comunalitati -> varianta exprimata in comun
r2 = r_XC * r_XC
# se obtin prin insumarea lui r patrat pe linii
comm = np.cumsum(r2, axis = 1)

t_comm = salvare_ndarray(
    comm,
    variabile_observate,
    "Indicatori",
    set_date.index.name,
    "data_out/Comm.csv",
)

corelograma(
    t_comm,
    "Comonalitati",
    0,
    "Reds",
    n < 15
)

# Trasare harta
gdf = GeoDataFrame.from_file("RO_NUTS/Ro.cpg")
print(gdf)

show()
