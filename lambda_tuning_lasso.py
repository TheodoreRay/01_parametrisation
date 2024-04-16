#source : https://www.kirenz.com/post/2019-08-12-python-lasso-regression-auto/#plot-values-as-a-function-of-lambda

#%% LIBRAIRIES 
import f_utilitaires as tools
import f_data_process as process
import f_plot as plot
import f_selection_de_features as FS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from importlib import reload
from sklearn.linear_model import Lasso

#%% IMPORT
data_init, _ = tools.Excel_to_DataFrame()
#%%
vturbine = data_init['ref_turbine_valorem'].unique()
#data = data_init[data_init['ref_turbine_valorem'] == 'T6'].sort_values(by = 'date_heure')[['temp_roul_gene2', 'temp_roul_gene1', 'puiss_active_produite', 'temp_stator']].dropna()
data = data_init[data_init['ref_turbine_valorem'] == 'T2'].sort_values(by = 'date_heure').dropna()

#%% AFFICHAGE DE L'INFLUENCE DE LAMBDA
plt.close('all')
reload(plot)
plot.cross_corr(data, 'temp_roul_gene2', 0)

#%% CONFIG
X_train = data[['temp_roul_gene1', 'puiss_active_produite', 'temp_stator']].iloc[:int(0.8*len(data)), :]
#X_train = data.drop(columns = ['Unnamed: 0', 'ref_turbine_valorem', 'temp_roul_gene2', 'date_heure']).iloc[:int(0.8*len(data)), :]
y_train = data['temp_roul_gene2'].iloc[:int(0.8*len(data))]
X_test = data[['temp_roul_gene1', 'puiss_active_produite', 'temp_stator']].iloc[int(0.8*len(data)):, :]
#X_test = data.drop(columns = ['Unnamed: 0', 'ref_turbine_valorem', 'temp_roul_gene2', 'date_heure']).iloc[int(0.8*len(data)):, :]
y_test = data['temp_roul_gene2'].iloc[int(0.8*len(data)):]

lambdas = [0.001, 0.01, 0.1, 0.5, 1, 2, 10]
coeff_a = np.zeros((len(lambdas), len(X_train.columns)))
train_r_squared = np.zeros(len(lambdas)); test_r_squared = np.zeros(len(lambdas))
train_rmse = np.zeros(len(lambdas)); test_rmse = np.zeros(len(lambdas))



#%% PLOT COEFF(LAMBDA) / R²(LAMBDA)
plt.close('all')
for ind, i in enumerate(lambdas):    
    reg = Lasso(alpha = i)
    reg.fit(X_train, y_train)
    coeff_a[ind,:] = reg.coef_
    const = reg.intercept_
    train_r_squared[ind] = reg.score(X_train, y_train)
    test_r_squared[ind] = reg.score(X_test, y_test)
    train_rmse[ind] = np.sqrt(mean_squared_error(y_train, X_train.dot(coeff_a[ind, :]) + const))
    test_rmse[ind] = np.sqrt(mean_squared_error(y_test, X_test.dot(coeff_a[ind, :]) + const))

plt.subplot(211)
plt.plot(train_r_squared, 'bo-', label=r'$R^2$ Training set', color="darkblue", alpha=0.6, linewidth=3)
plt.plot(test_r_squared, 'bo-', label=r'$R^2$ Test set', color="darkred", alpha=0.6, linewidth=3)
plt.xlabel('Lambda index'); plt.ylabel(r'$R^2$')
plt.xlim(0, 6)
plt.title(r'Evaluate lasso regression with lamdas: 0 = 0.001, 1= 0.01, 2 = 0.1, 3 = 0.5, 4= 1, 5= 2, 6 = 10')
plt.legend(loc='best')
plt.grid()
plt.subplot(212)
plt.plot(train_rmse, 'bo-', label='rmse Training set', color="darkblue", alpha=0.6, linewidth=3)
plt.plot(test_rmse, 'bo-', label='rmse Test set', color="darkred", alpha=0.6, linewidth=3)
plt.xlabel('Lambda index'); plt.ylabel(r'coeffs')
plt.xlim(0, 6)
plt.legend(loc='best')
plt.grid()
plt.figure()
#for i in range(len(X_train.columns)): plt.plot(coeff_a[:, i], label=X_train.columns[i], alpha=0.6, linewidth=3)
plt.plot(coeff_a[:, 0], 'bo-', label='coeff temp_roul_gene1', alpha=0.6, linewidth=3)
plt.plot(coeff_a[:, 1], 'ro-', label='coeff puiss_active_produite', alpha=0.6, linewidth=3)
plt.plot(coeff_a[:, 2], 'go-', label='coeff temp_stator', alpha=0.6, linewidth=3)
plt.xlabel('Lambda index'); plt.ylabel(r'coeffs')
plt.xlim(0, 6)
plt.title(r'Evaluate lasso coefficients with lamdas: 0 = 0.001, 1= 0.01, 2 = 0.1, 3 = 0.5, 4= 1, 5= 2, 6 = 10')
plt.legend(loc='best')
plt.grid()

#%% IDENTIFICATION DU MEILLEUR LAMBDA
df_lam = pd.DataFrame(test_r_squared, columns=['R_squared'])
df_lam['lambda'] = (lambdas)
Rc_best = df_lam.loc[df_lam['R_squared'].idxmax()]
Rc_actu = df_lam.loc[4, 'R_squared']
print(f'pour lambda = {Rc_best[1]}, gain de {Rc_best[0]-Rc_actu} par rapport à lambda=1')

df_rmse = pd.DataFrame(test_rmse, columns=['rmse'])
df_rmse['lambda'] = (lambdas)
rmse_best = df_rmse.loc[df_rmse['rmse'].idxmin()]
rmse_actu = df_rmse.loc[4, 'rmse']
print(f'pour lambda = {rmse_best[1]}, gain de {rmse_actu - rmse_best[0]} par rapport à lambda=1')