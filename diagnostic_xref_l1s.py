#%% LIBRAIRIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import traitement_parametrisation as f

#%% IMPORT
composant = 'refroidissement convertisseur'
turbine = 'T2'
modele_turbine = 'N131'
parc = 'ABLAI'
df1, _ = f.import_data_auto(format='parquet') #$\overline{\epsilon}$
df2, _ = f.import_data_auto(format='parquet') #ecart_mediane (double standardisation !)$
df_nan, _ = f.import_data_auto(format='parquet')
df_alert = f.import_data_auto(format='excel')
filepath = f'../../1_data/15_mean_std_indicators/{modele_turbine}_{parc}.xlsx'
dct_mean_std_epsi = pd.read_excel(filepath, composant, header=[0], index_col=[0])

#%% ls1 : analyse KDE l1 H1 (alert only) H0 (non-alert only)
df_alert1 = pd.DataFrame()
df_alert2 = pd.DataFrame()

## H1: alert only ##
for d in range(len(df_alert)):
    df_alert1 = pd.concat([df_alert1, df1[df_alert['début'][d]: df_alert['fin'][d]]])
    df_alert2 = pd.concat([df_alert2, df2[turbine][df_alert['début'][d]: df_alert['fin'][d]]])
df_alert1 = df_alert1.squeeze() #conversion Series
df_alert2 = df_alert2.squeeze() #conversion Series

## H0: non-alert only ##
"""df_alert1 = df1[df1.index.difference(df_alert1.index)]
df_alert2 = df2[df2.index.difference(df_alert2.index)]#"""

d_wass1 = f.wasserstein_daily(df_alert1)
d_wass2 = f.wasserstein_daily(df_alert2)

plt.subplot(211)
plt.title('répartition distance Wasserstein $\overline{\epsilon}$')
d_wass1.plot.kde(ind = np.linspace(d_wass1.min(), d_wass1.max(), len(d_wass1)), color='r', label='$H_1$ (alert only)')
plt.vlines(1.23, 0, 10, linestyle='dashed', color='b')
plt.grid(visible=True), plt.legend()
plt.subplot(212)
plt.title('répartition distance Wasserstein $\epsilon_{md}(y)$')
d_wass2.plot.kde(ind = np.linspace(d_wass2.min(), d_wass2.max(), len(d_wass2)), color='r', label='$H_1$ (alert only)')
plt.vlines(1.23, 0, 10, linestyle='dashed', color='b')
plt.grid(visible=True), plt.legend()

#%% xref : analyse KDE ecart_median journalier H0
plt.close('all')
plt.title('distributions journalières des écarts à la médiane hors-alerte')
for d in df1.index[::144]:
    df1['T4'][str(d)[:-9]: str(d)[:-9]].plot.kde(ind = np.linspace(df1['T4'][str(d)[:-9]: str(d)[:-9]].min(), df1['T4'][str(d)[:-9]: str(d)[:-9]].max(), len(df1['T4'][str(d)[:-9]: str(d)[:-9]])))
plt.grid(visible=True)