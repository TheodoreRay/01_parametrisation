#%% LIBRAIRIES
from tkinter import filedialog, Tk
import pandas as pd
import os
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

#%% FONCTIONS
def import_data_auto(format):
    # ouverture de la fenêtre d'intéraction
    root = Tk()
    root.destroy()
    root.mainloop()
    import_file_path = filedialog.askopenfilename()
    # obtention du nom du fichier
    filename = os.path.basename(import_file_path)

    # création du DataFrame
    if format=='parquet':
        df_data = pd.read_parquet(import_file_path)
    if format=='excel':
        df_data = pd.read_excel(import_file_path)    
    #df = SQLContext.read.csv("location", import_file_path)
    return df_data, filename

def wasserstein(data, title, do_plot): 
    # mise en forme Pandas #
    data = data.dropna()
    data_ref = pd.Series(data=np.random.normal(2.5e-1, 3e-1, len(data)))
    
    # calcul distance Wasserstein #
    if (np.abs(data.values)==0).sum() < len(data.dropna()):
        distance = scipy.stats.wasserstein_distance(data_ref.dropna(), data.dropna())
        x = np.linspace(min(list(data) + list(data_ref)), max(list(data) + list(data_ref)), len(data))
        
        # affichage #
        if do_plot:
            plt.figure(title)
            data_ref.plot.kde(ind = x)
            data.plot.kde(ind = x)
            plt.title(f'distance={distance}')
            plt.legend(), plt.grid()#"""
    else:
        distance = np.nan
    return distance

def wasserstein_daily(data):
    d_wass = pd.Series(name='distance W', dtype=object)
    for d in data.index[::144]:
        d_w = wasserstein(data[str(d)[:-9]: str(d)[:-9]], str(d), False)
        d_wass[str(d)[:-9]] = d_w
    return d_wass