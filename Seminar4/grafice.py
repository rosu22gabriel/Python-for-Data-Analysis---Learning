import matplotlib.pyplot as plt
import numpy as np

def plot_varianta(alpha:np.ndarray, procent_minimal=80, scal=True):
    n = len(alpha)
    x= np.arange(1, n+1)
    f = plt.figure(figsize=(8,5))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title("Plot varianta", color = "b", fontsize = 18)
    ax.set_xlabel("Componenta", fontsize=12)
    ax.set_ylabel("Varianta", fontsize=12)
    ax.set_xticks(x)
    ax.plot(x, alpha)
    ax.scatter(x, alpha, c="r", alpha=0.5)
    if scal:
        k1 = len(np.where(alpha > 1)[0] )
        ax.axvline(k1, c="g", label="Criteriul Kaiser")
    procent_cumulat = np.cumsum( alpha*100/sum(alpha) )
    k2 = np.where( procent_cumulat>procent_minimal )[0][0]+1
    ax.axvline(k2, c="c", label="Procent Minimal ("+str(procent_minimal)+")")
    ax.legend()
    return k1, k2


def show():
    plt.show()