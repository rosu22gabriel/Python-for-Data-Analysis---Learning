import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def plot_ierarhie(h:np.ndarray,etichete=None,color_threshold = 0,titlu="Plot Ierarhie"):
    f = plt.figure(titlu,figsize=(10,6))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu,fontdict={"fontsize":16})
    dendrogram(h,color_threshold=color_threshold,labels=etichete,ax=ax)
    if color_threshold!=0:
        ax.axhline(color_threshold,c="r")

def show():
    plt.show()