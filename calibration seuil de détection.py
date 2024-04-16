#%% LIBRAIRIES
import traitement_parametrisation as f
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from importlib import reload
from sklearn.model_selection import KFold
pd.options.mode.chained_assignment = None

#%% 1)
début = ['01-01-2021']#, '01-01-2021', '01-01-2021', '01-01-2021', '01-01-2021', '01-01-2021', '01-01-2021']
fin = ['01-01-2023']#, '07-01-2022', '07-01-2022', '07-01-2022', '07-01-2022', '07-01-2022', '07-01-2022']#, '11-01-2022']
parc = ['SOULA']
modele_turbine = ['G87']
df_H0H1HL = pd.read_excel('C:/Users/tra/JDrive/INTERVAL/Espace Datarooms/THESE THEODORE RAYMOND/3_excel/parcs.xlsx')
HL = list(df_H0H1HL[df_H0H1HL['NOM_PROJET_CODE']==parc[0]].loc[:, 'HL[0]':'HL[1]'].values[0])# AAAA-MM-JJ # période d'apprentissage
H0 = list(df_H0H1HL[df_H0H1HL['NOM_PROJET_CODE']==parc[0]].loc[:, 'H0[0]':'H0[1]'].values[0])# AAAA-MM-JJ # période d'apprentissage
data, dct_models, header = f.import_data(v_parc=parc, v_modele_turbine=modele_turbine, v_début=début, v_fin=fin, is_learning_data=True) 
dct_models = dct_models[parc[0]]
data_HL = {'brutes':0, 'standardisées':0}
data_H0 = {'brutes':0, 'standardisées':0}
data_HL['brutes'] = data[parc[0]][(data[parc[0]]['date_heure']>=HL[0]) & (data[parc[0]]['date_heure']<=HL[1])]
data_H0['brutes'] = data[parc[0]][(data[parc[0]]['date_heure']>=H0[0]) & (data[parc[0]]['date_heure']<=H0[1])]
data_HL['standardisées'] = data_HL['brutes'].copy()
data_H0['standardisées'] = data_H0['brutes'].copy()
v_turbine = list(data_HL['brutes']['ref_turbine_valorem'].unique())
## STANDARDISATION ##
[data_HL['standardisées'], data_H0['standardisées']], _, _ = \
        f.standardisation_data(data_HL['brutes'], [data_HL['standardisées'], data_H0['standardisées']], v_turbine)
s_composant = pd.Series(index = list(dct_models.keys())[:-2], data = [dct_models[composant].index.names[1] for composant in list(dct_models.keys())[:-2]])
if header[parc[0]]['modele_turbine'] in ['G58', 'G87', 'G90', 'G97', 'G114', 'N80', 'N117', 'N131']: 
	s_composant['roulement 1 génératrice'] = 'temp_stator'; s_composant['roulement 2 génératrice'] = 'temp_rotor'

# ANALYSE DE SEUILS DE DETECTION 5%
reload(f); plt.close('all')
#N = 20
labels = ['$\epsilon_{X_0}$']
turbines = v_turbine
composants = ['palier arbre lent']
tsi = dict((composant, dict((label, pd.DataFrame(columns = turbines)) for label in labels) ) for composant in s_composant.keys())
tsi_H0 = dict((composant, dict((label, pd.DataFrame(columns = turbines)) for label in labels) ) for composant in s_composant.keys())
id_nan = dict((composant, dict((label, dict((turbine, pd.Series(dtype=float)) for turbine in turbines)) for label in labels) ) for composant in s_composant.keys())

for composant in composants:
    print(composant)
    for label in labels:
        print(label)
        if label == '$\epsilon_{md}$':
            tsi[composant][label], _ = f.ecart_mediane(v_turbine, data_HL['standardisées'], s_composant[composant], HL)
        elif label == '$\overline{\epsilon}$':
            for turbine in v_turbine:
                tsi[composant][label][turbine] = pd.concat([tsi[composant][label][turbine] for label in labels], axis=1).mean(axis=1, numeric_only=True)
        else:
            régresseurs = list(dct_models[composant].loc[label].index)
            # apprentissage du modèle #
            df_coef, v_MAE, YX_train, YX_test = f.model_learning(data_HL['standardisées'], v_turbine, s_composant[composant], régresseurs, 'least squares', True)
            _, tsi[composant][label] = f.residu_mono_multi(data_HL['standardisées'], s_composant[composant], régresseurs, v_turbine, df_coef)
            _, tsi_H0[composant][label] = f.residu_mono_multi(data_H0['standardisées'], s_composant[composant], régresseurs, v_turbine, df_coef)
            for turbine in v_turbine:
                mu = tsi[composant][label][turbine][tsi[composant][label][turbine].index.difference(id_nan[composant][label][turbine])].mean()
                std = tsi[composant][label][turbine][tsi[composant][label][turbine].index.difference(id_nan[composant][label][turbine])].std()
                id_nan[composant][label][turbine] = tsi[composant][label][turbine][tsi[composant][label][turbine].isna()].index
                tsi[composant][label][turbine] = (tsi[composant][label][turbine]).ewm(alpha=0.01).mean()
                tsi[composant][label][turbine] = (tsi[composant][label][turbine]-mu) / std
                tsi_H0[composant][label][turbine] = (tsi_H0[composant][label][turbine]).ewm(alpha=0.01).mean()
                tsi_H0[composant][label][turbine] = (tsi_H0[composant][label][turbine]-mu) / std
            tsi[composant][label][turbine] = tsi[composant][label][turbine].sort_index()

# données de post traitement ##
# mesure des couples (N2I, tau) #
plt.close('all')
reload(f)
composant = composants[0]
label = labels[0]#labels#'$\epsilon_{X_0}$'
dépendance = v_turbine
dct_xy = dict((d, 0) for d in dépendance)#index = [round(x,2) for x in list(np.arange(0, 3.0, 0.1))], columns = dépendance)
df_tau = pd.DataFrame(index=dépendance, columns=['N2I', '%'], dtype=float)
start_time = time.time()
for d in ['T1']:#dépendance:
    print(d)
    tau_5p = round(np.percentile(tsi[composant][label][d][tsi[composant][label][d].index.difference(id_nan[composant][label][d])].sort_index(), 95), 2)
    tsi_5p = tsi[composant][label][d][tsi[composant][label][d]>tau_5p]
    dct_xy[d] = pd.Series(index=pd.Index([round(x,2) for x in list(np.arange(tsi_5p.max(), tsi_5p.min(), -0.2))]), name='N2I', dtype=float)
    print(len(dct_xy[d].index))
    for tau in [0.8]:#dct_xy[d].index:
        plt.figure(f'{tau}, {d}')
        dates, dct_xy[d][tau] = f.N2I(tsi_5p, tau)
        print(dates)

print(f'{time.time()-start_time} pour calibrer 6 seuils')

# affichages ##
color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
# courbes N2I(tau) #
plt.figure(f'N2I(tau)')
for x, d in enumerate(dépendance):
    plt.plot(dct_xy[d], marker='.', label=d, color=color[x])
    plt.plot(df_tau.loc[d, 'N2I'], dct_xy[d][df_tau.loc[d, 'N2I']], marker='.', markersize=20, color=color[x])
plt.hlines(1, 0, 3.0, 'k', linestyle='dashed')
plt.xlabel('tau'), plt.ylabel('N2I')
plt.grid(), plt.legend(), plt.title(f'N2I(tau) {parc[0]} par turbine')

#%% 2)
# import de l'erreur de prédiction
data, filename = f.Excel_to_DataFrame()
data = data.iloc[:, 1:]

# SI PAS DE DEFAUT
turbine = 'T6'
plt.close('all')
vturbine = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
vstd = []; kf = KFold(n_splits = 20)
max_y = []; max_s = []; min_y = []; min_s = []

for turbine in vturbine:
    
    # définition de la plage de seuil
    #for train, test in kf.split(data[turbine]):
        #vstd.append(data[turbine][test].std())
    #std = np.median(vstd)
    std = data[turbine].std()
    range_s = np.arange(std*0.1, 5*std, 0.2*std)
    
    # affichage résidu
    plt.subplot(211)
    plt.plot(data[turbine], label = turbine)
    
    # établissement du taux de faux positifs par seuil (axe y)
    y = []
    for i, s in enumerate(range_s):
        FP = len(data[data[turbine] > s+data[turbine].mean()]) #taux de faux positifs
        y.append((FP/len(data[turbine]))*100)
    
    # affichage pseudo courbe ROC
    plt.subplot(212)
    plt.title('évolution du taux de faux positifs en fonction de la valeur du seuil')
    plt.plot(range_s, y, 'o-', label = turbine), plt.grid()
    max_y.append(np.max(y)); max_s.append(range_s[-1])
    min_y.append(np.min(y)); min_s.append(range_s[0])
    
plt.grid()
plt.hlines(1, min(min_s), max(max_s))
plt.yticks(ticks = np.arange(min(min_y), max(max_y), 5), label =- np.arange(min(min_y), max(max_y), 5))
plt.xticks(ticks = np.arange(min(min_s), max(max_s), max(max_s)/20), label = np.arange(min(min_s), max(max_s), max(max_s)/20), rotation=35)
plt.xlabel('seuil (°C)'); plt.xlim(min(min_s), max(max_s))
plt.ylabel('taux de faux positifs (%)'); plt.ylim(min(min_y), max(max_y))
plt.legend()

# ROUTINE SI DEFAUT
H1 = [data['date_heure'][0], data['date_heure'][6170]] #data[data['date_heure']<H0[1]]
H0 = [data['date_heure'][6170], data['date_heure'][33493]]

plt.close('all')

data_H0 = data[data['date_heure'] > H0[0]]
data_H1 = data[data['date_heure'] < H1[1]]
t = data['date_heure'].values; x = []; y1 = []; y2 = []; indice = 15
range_s = np.arange(data_H0['erreur de prédiction'].std()*0.1, 5*data_H0['erreur de prédiction'].std(), 0.2*data_H0['erreur de prédiction'].std())

plt.plot(t, data['erreur de prédiction']), plt.grid(), plt.title('erreur de prédiction') 
plt.ylim(0, max(data['erreur de prédiction']))
plt.xlim(t[0], t[-1])
for i, s in enumerate(range_s):
    première_detection = data_H1['date_heure'][data_H1['erreur de prédiction'] > s+data_H1['erreur de prédiction'][:2000].mean()].iloc[0]
    TP = len(data_H1[data_H1['erreur de prédiction'] > s+data_H1['erreur de prédiction'][:2000].mean()]) #taux de vrais positifs
    FP = len(data_H0[data_H0['erreur de prédiction'] > s+data_H0['erreur de prédiction'][:2000].mean()]) #taux de faux positifs
    AVD = len(data_H1[data_H1['date_heure'] > première_detection])
    x.append((FP/len(data_H0))*100)
    y2.append(int((TP/len(data_H1))*100))
    y1.append(int(AVD/144))
    if i == indice: #mise en valeur du seuil opti
        print(FP), print(TP)
        plt.hlines(y = s+data_H1['erreur de prédiction'][:2000].mean(), xmin=t[0], xmax=t[len(data_H1)], colors='r')
        plt.hlines(y = s+data_H0['erreur de prédiction'][:2000].mean(), xmin=t[len(data_H1)], xmax=t[len(data_H0)+len(data_H1)], colors='r')
        print(AVD)
    else:
        plt.hlines(y = s+data_H1['erreur de prédiction'][:2000].mean(), xmin=t[0], xmax=t[len(data_H1)], colors='grey', linestyles='dashed')
        plt.hlines(y = s+data_H0['erreur de prédiction'][:2000].mean(), xmin=t[len(data_H1)], xmax=t[len(data_H0)+len(data_H1)], colors='grey', linestyles='dashed')  

plt.figure()
plt.title(f'ROC {round(range_s[0], 2)}:{round(range_s[-1], 2)} (lecture de droite à gauche)')
plt.plot(x, y1, 'o-'), plt.grid()
plt.xticks(ticks=np.arange(min(x), max(x), 2), label=np.arange(min(x), max(x), 2), rotation=35)
plt.xlabel('taux de faux positifs (%)')
plt.ylabel('avance à la détection (Jours)')

plt.figure()
plt.title(f'ROC {round(range_s[0], 2)}:{round(range_s[-1], 2)} (lecture de droite à gauche)')
plt.plot(x, y2, 'o-'), plt.grid()
plt.yticks(ticks=np.arange(min(y2), max(y2), 2), label=np.arange(min(y2), max(y2), 2), rotation=35)
plt.xticks(ticks=np.arange(min(x), max(x), 2), label=np.arange(min(x), max(x), 2), rotation=35)
plt.xlabel('taux de faux positifs (%)')
plt.ylabel('taux de vrais positifs (%)')