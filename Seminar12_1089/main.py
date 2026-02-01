import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import f

from functii import calcul_metrici, salvare_ndarray
from grafice import f_distributii, show, f_scatter

t = pd.read_csv("data_in/Hernia/hernia.csv",index_col=0)
variabile = list(t)
tinta = variabile[-1]
predictori = variabile[:-1]

# Splitare in invatare si testare
t_train,t_test,y_train,y_test = train_test_split(
    t[predictori],t[tinta],test_size=0.3
)

# Evaluare predictori
x = t_train[predictori].values
x_ = np.mean(x,axis=0)

model_lda = LinearDiscriminantAnalysis()
model_lda.fit(t_train,y_train)

g = model_lda.means_
n = len(t_train)
ponderi = model_lda.priors_
q = len(ponderi)
dg = np.diag(ponderi)*n
# print(dg)
#Imprastiere totala
sst = (x-x_).T@(x-x_)
# Imprastiere interclasa
ssb = (g-x_).T@dg@(g-x_)
ssw = sst-ssb
# Putere discriminare
f_predictori = (np.diag(ssb)/(q-1))/(np.diag(ssw)/(n-q))
pvalues = 1 - f.cdf(f_predictori,q-1,n-q)

t_predictori = pd.DataFrame(
    data={
        "Putere discriminare":f_predictori,
        "PValues":pvalues
    }, index=predictori
)
t_predictori.to_csv("data_out/Predictori.csv")

clase = model_lda.classes_
for predictor in predictori:
    f_distributii(t_train,predictor,y_train,clase)

# Testare model pe setul de test
predictie_test_lda = model_lda.predict(t_test)
t_predictii_test = pd.DataFrame(
    data={
        tinta:y_test,
        "Predictie LDA":predictie_test_lda
    }, index=t_test.index
)

t_predictii_test.to_csv("data_out/Predictii_test.csv")

metrici_lda = calcul_metrici(y_test,predictie_test_lda,clase)
metrici_lda[0].to_csv("data_out/CM_lda.csv")
metrici_lda[1].to_csv("data_out/Acuratete_lda.csv")

# Analiza scorurilor discriminante
z = model_lda.transform(t_train)
etichete_z = ["Z"+str(i+1) for i in range(q-1)]
t_z = salvare_ndarray(
    z,t_train.index,etichete_z,"data_out/z.csv"
)
assert isinstance(t_z,pd.DataFrame)
t_gz = t_z.groupby(by=y_train).mean()
gz = t_gz.values
f_scatter(t_z,t_gz,y_train,clase)
z_ = np.mean(z,axis=0)

sst_z = (z-z_).T@(z-z_)
ssb_z = (gz-z_).T@dg@(gz-z_)
ssw_z = sst_z-ssb_z
f_z = (np.diag(ssb_z)/(q-1))/(np.diag(ssw_z)/(n-q))
pvalues_z = 1 - f.cdf(f_z,q-1,n-q)
print(f_z)
print(pvalues_z)

show()
