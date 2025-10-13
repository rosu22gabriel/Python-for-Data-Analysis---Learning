import pandas as pd
from pandas.api.types import is_numeric_dtype

def nan_replace_df(t:pd.DataFrame):
    for c in t.columns:
        if any(t[c].isna()):
            if is_numeric_dtype(t[c]):
                # t[c].fillna(t[c].mean(), inplace=True)
                t.fillna({c: t[c].mean()}, inplace=True)
            else:
                t.fillna({c:t[c].mode()[0]}, inplace=True)


def calcul_ponderi(t:pd.Series):
    return t/t.sum()

def diversitate(t:pd.Series):
    #
    return pd.Series([], ["Shannon", "Simpson", "InvSimpson"])