import numpy as np
import pandas as pd

def nan_replace(x: np.ndarray):
    is_nan = np.isnan(x)
    print(type(is_nan))
    k = np.where(is_nan)