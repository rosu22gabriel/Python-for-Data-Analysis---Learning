import pandas as pd

from functii import nan_replace_df, tabelare_varianta
from functii import acp
from grafice import plot_varianta
from grafice import show

set_date = pd.read_csv("data_in/Teritorial2022/Teritorial_2022.csv", index_col=0)
nan_replace_df(set_date)

variabile_observate = list(set_date)[3:]
x = set_date[variabile_observate].values
# acp(x)
alpha, a = acp(x)

#Analiza variantei componentelor
t_varianta = tabelare_varianta(alpha)
t_varianta.round(3).to_csv("data_out/Varianta.csv")
plot_varianta(alpha)
show()

