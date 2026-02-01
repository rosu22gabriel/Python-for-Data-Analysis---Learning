import matplotlib.pyplot as plt
import pandas as pd
from seaborn import kdeplot,scatterplot

def f_distributii(t:pd.DataFrame,variabila,y,clase,titlu="Distributii"):
    f = plt.figure(titlu+" - "+variabila,figsize=(10,6))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu+" - "+variabila,fontdict={"fontsize":16})
    kdeplot(t,x=variabila,hue=y,hue_order=clase,ax=ax,fill=True)

def f_scatter(t:pd.DataFrame,tg:pd.DataFrame,y,clase,
              varx="Z1",vary="Z2",titlu="Plot instante in axe discriminante"):
    f = plt.figure(titlu+" - "+varx+" "+vary,figsize=(10,6))
    ax = f.add_subplot(1,1,1)
    ax.set_title(titlu,fontdict={"fontsize":16})
    scatterplot(t,x=varx,y=vary,hue=y,hue_order=clase,ax=ax)
    scatterplot(tg,x=varx,y=vary,hue=clase,hue_order=clase,legend=False,
                ax=ax,marker = "s", s = 100)


def show():
    plt.show()

