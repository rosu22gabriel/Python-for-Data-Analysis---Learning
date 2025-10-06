import sys
import numpy as np
import pandas as pd

from functii import nan_replace

# In fisierul Teritorial_2022.csv sunt date privind indicatori
# statistici din diverse domenii si subdomenii economico-sociale.
# Sa se implementeze urmatoarele cerinte:

# 1.Inlocuirea valorilor lipsa cu media

# 2.Standardizarea si centrarea datelor

# 3.Calculul matricelor de corelatie si covarianta totale si pe grupe
# (daca exista criteriu de grupare)

pd.set_option("display.max_columns", None)
np.set_printoptions(6, threshold=sys.maxsize, suppress=True)

tabel_date = pd.read_csv("data_in/Teritorial_2022.csv", index_col = 0)

# print(type(tabel_date))
# print(tabel_date)

variabile_numerice = list(tabel_date.columns[3:]) #[start:end:step]
# print(variabile_numerice, type(variabile_numerice))

x = tabel_date[variabile_numerice].values
# print(type(x))
# print(x)

# 1.
nan_replace(x)
print(np.where(np.isnan(x)))
