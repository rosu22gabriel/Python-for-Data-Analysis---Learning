import pandas as pd
import numpy as np
import sklearn.cross_decomposition as skl
import matplotlib.pyplot as plt

ind_df = pd.read_csv(filepath_or_buffer="./dataIN/Industrie.csv", index_col=0)
#aceeasi prima coloana in fisiere si se face merge pe index
#print(ind_df.head())

pop_df = pd.read_csv(filepath_or_buffer='./dataIN/PopulatieLocalitati.csv', index_col=0)

#print(pop_df)

# Cerinta 1 ---------------------------------
merge_df = ind_df.merge(right=pop_df, left_index=True, right_index=True) #merge pe index, de aici rezulta coloana de localitate de doua ori=> e nevoie de o lista de coloane pe care dorim sa le selectam
#print(merge_df)

# creare lista coloane cu industrii
industrii_list = ind_df.columns[1:].values.tolist()
#print(industrii_list, type(industrii_list))

def perCapita(df, vars, pop):
    linie = df[vars] / df[pop]
   # print(linie) # => pt fiecare localitate se obtine impartirea CA la populatie
    rez = list(linie)
    rez = [df['Localitate_x']]+ rez
    return pd.Series(data=rez, index=['Localitate']+vars)

rezultat_df = merge_df[['Localitate_x', 'Populatie'] + industrii_list].\
    apply(func=perCapita, axis=1, vars=industrii_list, pop='Populatie') # axa pe linii, parametrii spun functiei ce coloane folosim din df merge

#print(rezultat_df)
#siruta e acum index, dar vrem sa l salvam ca si coloana
rezultat_df.to_csv('./dataOUT/Cerinta1.csv')

# Cerinta 2 ----------------------------------
#sume pe col cu industrii => cifre de afaceri cumulate si unde e cea mai mare se ia industria

df_1 = merge_df[industrii_list + ['Judet']].groupby(by='Judet').sum()
#print(df_1)
def indDominanta(df):
    linie = df.values
    # print(linie)
    maxCA = np.argmax(linie) # => index valoare max
    return pd.Series(data= [df.index[maxCA], linie[maxCA]], index=['Industria dominanta', 'Cifra afaceri'])

df_2 = df_1[industrii_list].apply(func=indDominanta, axis=1)

df_2.to_csv('./dataOUT/Cerinta2.csv', index_label='Cod judet')

# Cerinta 3 ------------------------------------
tabel = pd.read_csv(filepath_or_buffer='./dataIN/DataSet_34.csv', index_col=0)

print(tabel)
obs = tabel.index.values
def standardizare(A):
    #asteapta un numpy.ndarray
    medii = np.mean(A, axis=0) #medii pe col
    abateri_std= np.std(a=A, axis=0)
    return (A- medii)/abateri_std

x_col = tabel.columns[0:4].values
print(x_col)
X = tabel[x_col].values
Xstd = standardizare(X)
Xstd_df =pd.DataFrame(data=Xstd, index= obs, columns = x_col )
Xstd_df.to_csv('./dataOUT/Xstd.csv', index_label = 'Tara')

y_col= tabel.columns[4:].values
print(y_col)
Y = tabel[y_col].values
Ystd = standardizare(Y)
Ystd_df =pd.DataFrame(data=Ystd, index= tabel.index.values, columns = y_col )
Ystd_df.to_csv('./dataOUT/Ystd.csv', index_label = 'Tara')

# Cerinta 4 --------------------------------
n,p=Xstd.shape #n-nr linii/observatiilor/tarilor; p-nr coloane
print(n,p) #n=27, p=4

q=Ystd.shape[1]
print(q) #q=4

#nr de perechi canonice = minimul intre p si q
m = min(p,q)

obiect_CCA = skl.CCA(n_components=m) #canonical correlation analysis
obiect_CCA.fit(X=Xstd, y=Ystd)
z,u = obiect_CCA.transform(X=Xstd, y=Ystd) #returneaza 2 param z si u
#matrici de sol cu 27 de linii
#print(z)
#print(u)
z_df = pd.DataFrame(data = z, index = obs, columns=x_col)
z_df.to_csv('./dataOUT/Xscores.csv', index_label='Tara')

u_df = pd.DataFrame(data = u, index = obs, columns=y_col)
u_df.to_csv('./dataOUT/Yscores.csv', index_label='Tara')

#Cerinta 5 -------------------------------------------
#corelatiile factoriale
Rxz = obiect_CCA.x_loadings_
print(Rxz) #linie=var obs, coloane= canonice
Rxz_df = pd.DataFrame(data=Rxz, index=x_col, columns = ('z'+str(j+1) for j in range(m)))
Rxz_df.to_csv('./dataOUT/Rxz.csv')

Ryu = obiect_CCA.y_loadings_
print(Ryu) #linie=var obs, coloane= canonice
Ryu_df = pd.DataFrame(data=Ryu, index=y_col, columns = ('u'+str(j+1) for j in range(m)))
Ryu_df.to_csv('./dataOUT/Ryu.csv')

# Cerinta 6 --------------------------------
#coord punctelor sunt prodporc si prodVite din Xstd si Ystd
def biplot(z, u, titlu= 'Biplot', etichetaX='z1, u1', etichetaY = 'z2, u2',obs=None):
    plt.figure(num=titlu, figsize = (15,10))
    plt.title(label=titlu, fontsize=12, verticalalignment= 'bottom')
    plt.xlabel(xlabel=etichetaX, fontsize=10, verticalalignment='top')
    plt.ylabel(ylabel=etichetaY, fontsize=10, verticalalignment='bottom')
    plt.scatter(x=z[:,0],y= z[:,1], color = 'Red', label= 'Set X')
    plt.scatter(x=u[:,0],y= u[:,1], color = 'Blue', label = 'Set Y')
    if obs is not None:
        for i in range(len(obs)):
            plt.text(x=z[i,0], y=z[i,1], s=obs[i])
            plt.text(x=u[i,0], y=u[i, 1], s=obs[i])
    plt.legend()


biplot(z=z,u=u, obs= obs)
plt.show()

