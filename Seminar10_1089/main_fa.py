import sys

import numpy as np
import pandas as pd

from functii import nan_replace_df, salvare_ndarray, tabelare_varianta_factori
from grafice import plot_varianta, show, corelograma, scatterplot, plot_harta
from geopandas import GeoDataFrame

import factor_analyzer as fa

np.set_printoptions(3,sys.maxsize,suppress=True)
pd.set_option("display.max_columns",None)

set_date = pd.read_csv("data_in/Teritorial2022/Teritorial_2022.csv",index_col=0)
nan_replace_df(set_date)

variabile_observate = list(set_date)[3:]
x = set_date[variabile_observate].values
n,m = x.shape

# Analiza factorabilitatii
# Testul Bartlett
test_bartlett = fa.calculate_bartlett_sphericity(x)
# print(test_bartlett)
if test_bartlett[1]>0.001:
    print("Nu exista factori comuni!")
    exit(0)
# Index KMO
kmo = fa.calculate_kmo(x)
# print(kmo)
t_kmo = pd.DataFrame(
    data={
        "KMO":np.append(kmo[0],kmo[1])
    },index=variabile_observate+["Total"]
)
t_kmo.to_csv("data_out_fa/kmo.csv")
corelograma(t_kmo,"Index KMO",0,"Greens")
# show()

# Construire model factorial
# metoda_rotatie = None
metoda_rotatie = "varimax"
model_fact = fa.FactorAnalyzer(m,rotation=metoda_rotatie)
model_fact.fit(x)

# Analiza variantei
# Varianta factorilor comuni
varianta = model_fact.get_factor_variance()
# print(varianta)
t_varianta = tabelare_varianta_factori(varianta)
t_varianta.round(3).to_csv("data_out_fa/Varianta.csv")
# Varianta specifica (varianta factorilor specifici)
psi = model_fact.get_uniquenesses()
t_psi = pd.Series(
    np.append(psi,sum(psi)),
    variabile_observate+["Total"]
)
t_psi.name="Varianta"
t_psi.to_csv("data_out_fa/Varianta_specifica.csv")
# print(psi,sum(psi))
# Comunalitati
comm = model_fact.get_communalities()
t_comm = pd.Series(
    np.append(comm,sum(comm)),variabile_observate+["Total"]
)
t_comm.name="Comunalitate"
corelograma(pd.DataFrame(t_comm),"Comunalitati",0,"Blues")
corelograma(pd.DataFrame(t_psi),"Varianta specifica",0,"Reds")
k = plot_varianta(varianta[0],70,eticheta_x="Factor")
nr_min_factori = min( [v for v in k if v is not None] )

print("Numar factori semnificativi:",nr_min_factori)

# Analiza corelatiilor factoriale (corelatii variabile-factori)
l = model_fact.loadings_
etichete_factori = list(t_varianta.index)
t_l = salvare_ndarray(
    l,
    variabile_observate,
    etichete_factori,
    "Indicatori",
    "data_out_fa/l.csv"
)
corelograma(t_l,"Corelograma - AFact",annot=m<=10)
for i in range(2,nr_min_factori+1):
    scatterplot(
        t_l,
        "F1",
        "F"+str(i),
        "Plot corelatii - AFact",
        t_l.index,
        True
    )

# Analiza scorurilor
f = model_fact.transform(x)
t_f = salvare_ndarray(
    f,
    set_date.index,
    etichete_factori,
    set_date.index.name,
    "data_out_fa/f.csv"
)
for i in range(2,nr_min_factori+1):
    scatterplot(
        t_f,
        "F1",
        "F"+str(i),
        "Plot scoruri - AFact",
        t_f.index
    )

# Trasare harta
gdf = GeoDataFrame.from_file("RO_NUTS2/Ro.shp")
# print(gdf)
camp_legatura = "sj"
for i in range(1,nr_min_factori+1):
    plot_harta(gdf,camp_legatura,t_f,"F"+str(i))

show()
