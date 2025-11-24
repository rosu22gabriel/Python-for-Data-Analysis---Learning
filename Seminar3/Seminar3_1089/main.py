import numpy as np
import pandas as pd

def medie_ponderata(t:pd.DataFrame):
    x = t.values
    m = np.average(x[:, :-1], axis=0, weights=x[:, -1])
    return pd.Series(m, t.columns[:1])


pd.set_option("display.max_columns",None)

industria_alimentara = pd.read_csv("data_in/IndustriaAlimentara.csv",index_col=0)

industrii = list(industria_alimentara)[1:]

numar_angajati = industria_alimentara[industrii].apply(func=lambda x:x.sum(),axis=1)
# print(numar_angajati)
cerinta1 = industria_alimentara[numar_angajati>0]
cerinta1.to_csv("data_out/Cerinta1.csv")

procent_angajati = industria_alimentara[industrii].apply(func=lambda x:x/x.sum(),axis=1)
assert isinstance(procent_angajati,pd.DataFrame)
procent_angajati.insert(0,"Localitate",industria_alimentara["Localitate"])
procent_angajati.fillna(0,inplace=True)
procent_angajati.to_csv("data_out/Cerinta2.csv")

populatia = pd.read_csv("data_in/PopulatieLocalitati.csv",index_col=0)
industria_alimentara_loc = industria_alimentara.merge(populatia,left_index=True,right_index=True)

pondere_angajati =pd.DataFrame(industria_alimentara_loc[industrii+["Populatie"]].
                    apply(func=lambda x:x[industrii].sum()/x["Populatie"],axis=1),columns=["PondereAngajati"])
pondere_angajati.insert(0,"Localitate",industria_alimentara_loc["Localitate_x"])
pondere_angajati.to_csv("data_out/pondere_angajati_loc.csv")


# Cerinta 3
industria_alimentara_judet = industria_alimentara_loc[industrii+["Populatie", "Judet"]].groupby(by="Judet").sum()
# print(industria_alimentara_judet)
assert isinstance(industria_alimentara_judet, pd.DataFrame)
cerinta3 = industria_alimentara_judet.apply(func=lambda x:x[industrii].sum()/x["Populatie"], axis=1)
cerinta3.name = "Pondere"
# print(cerinta3)
assert isinstance(cerinta3, pd.Series)
cerinta3.sort_values(inplace=True)
cerinta3.to_csv("data_out/Cerinta3.csv")

# Cerinta 4
print(industria_alimentara_judet)
cerinta4 = industria_alimentara_judet[industrii].apply(func=lambda x: x.index[x.argmax()], axis=1)
cerinta4.name = "Activitate"
cerinta4.to_csv("data_out/Cerinta4.csv")

print(industria_alimentara_loc)
# Cerinta 5
coduri_judete = pd.read_csv("data_in/Coduri_Judete.csv",index_col=0)
industria_alimentara_loc_reg = industria_alimentara_loc.merge(
    coduri_judete,
    left_on="Judet",
    right_index=True
)
# print(industria_alimentara_loc_reg)
cerinta5 = industria_alimentara_loc_reg[industrii+["Populatie","Regiune"]].groupby(by="Regiune").apply(
    func=medie_ponderata,
    include_groups=False)
# print(cerinta5)
cerinta5.to_csv("data_out/Cerinta5.csv")

# Cerinta 6
print(industria_alimentara_judet)
total_national = industria_alimentara_judet.sum(axis = 0)
# print(total_national)
cerinta6 = industria_alimentara_judet.apply(func=lambda x:x[industrii]/total_national[industrii]/( x["Populatie"]/total_national["Populatie"]),
                                            axis=1)
cerinta6.round(3).to_csv("data_out/Cerinta6.csv")