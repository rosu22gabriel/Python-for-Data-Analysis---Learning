import pandas as pd

pd.set_option("display.max_columns", None)

from functii import nan_replace_df, calcul_ponderi

t = pd.read_csv("data_in/Religious.csv", index_col=0)

nan_replace_df(t)
# print(t)
confesiuni = list(t)[2:]

coduri_localitati = pd.read_csv("data_in/Coduri_Localitati.csv", index_col=0)

t_localitate = t[confesiuni].merge(coduri_localitati["County"], left_index=True, right_index=True)

# print(t_localitate)

t_judet = t_localitate.groupby(by="County").sum()
t_judet.to_csv("data_out/Religious_County.csv")

coduri_judete = pd.read_csv("data_in/Coduri_judete.csv", index_col=0)

assert isinstance(t_judet, pd.DataFrame)
t_judet_ = t_judet.merge(coduri_judete["Regiune"], left_index=True, right_index=True)
t_regiune = t_judet_.groupby(by="Regiune").sum()
t_regiune.to_csv("data_out/Religious_Region.csv")

t_p_loc = t[confesiuni].apply(func=calcul_ponderi, axis=1)
assert isinstance(t_p_loc, pd.DataFrame)
t_p_loc.insert(0, "City", t["City"])

t_p_loc.to_csv("data_out/Religious_p_loc.csv")

t_p_judet = t_judet.apply(func=lambda x:x/x.sum(), axis=1)
t_p_judet.to_csv("data_out/Religious_p_county")