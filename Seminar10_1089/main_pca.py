import sys

import numpy as np
import pandas as pd

from functii import nan_replace_df, tabelare_varianta, salvare_ndarray
from grafice import plot_varianta, show, corelograma, scatterplot
from geopandas import GeoDataFrame

from sklearn.decomposition import PCA

np.set_printoptions(3,sys.maxsize,suppress=True)
pd.set_option("display.max_columns",None)

set_date = pd.read_csv("data_in/Teritorial2022/Teritorial_2022.csv",index_col=0)
nan_replace_df(set_date)

variabile_observate = list(set_date)[3:]
x = set_date[variabile_observate].values
n,m = x.shape

x_ = (x - np.mean(x,axis=0))/np.std(x,axis=0)

model_acp = PCA()
model_acp.fit(x_)

# Analiza varianta componente
alpha = model_acp.explained_variance_*(n-1)/n
# print(alpha)

# ....

c = model_acp.transform(x_)
print(c)

# Idem main.py

