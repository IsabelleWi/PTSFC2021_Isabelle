
#%%
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm
tqdm.pandas()

import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import trange

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from Seq_to_Seq_Functions import *
from Preprocessing_Evaluation_Functions import *
from Models_Functions import *

def Prepare_new_Input(date, 
                      month_number, 
                      wind = True, 
                      add_month = True):
  

  """

  date: String in Format YYYYMMDD
  wind: Boolean; True if Wind Data, False if Temp Data

  """

  if wind == True:

    new_forecast = pd.read_csv('icon-eu-eps_' + date + '00_wind_mean_10m_Berlin.txt', delimiter="|",comment='#')


  if wind == False:

    new_forecast = pd.read_csv('icon-eu-eps_' + date + '00_t_2m_Berlin.txt', delimiter="|",comment='#')


  new_forecast = new_forecast.iloc[:,1:-1]

  new_forecast["ens_mean"]=new_forecast.iloc[:,1:].mean(axis=1)
  new_forecast["ens_var"]=new_forecast.iloc[:,1:].var(axis=1)

  if add_month == True:

    m1 = pd.DataFrame(data = [[0]* (month_number - 1)]*65, columns = list(range(1,month_number)))
    m2 = pd.DataFrame(data = [[1]]*65, columns =[month_number])
    m3 = pd.DataFrame(data = [[0]* (12 - month_number)]*65, columns = list(range(month_number + 1,12 + 1)))
    month_dummies = pd.concat([m1, m2, m3], axis=1)

    new_forecast = pd.concat([new_forecast, month_dummies], axis =1)

  new_forecast = pd.concat([new_forecast.reset_index(drop = True), 
                    pd.DataFrame(list(new_forecast["ens_mean"].shift(6)), columns =["ens_mean_-6"]).reset_index(drop = True), 
                    pd.DataFrame(list(new_forecast["ens_mean"].shift(2)), columns =["ens_mean_-2"]).reset_index(drop = True), 
                    pd.DataFrame(list(new_forecast["ens_mean"].shift(1)), columns =["ens_mean_-1"]).reset_index(drop = True), 
                    pd.DataFrame(list(new_forecast["ens_mean"].shift(-1)), columns =["ens_mean_1"]).reset_index(drop = True), 
                    pd.DataFrame(list(new_forecast["ens_mean"].shift(-2)), columns =["ens_mean_2"]).reset_index(drop = True), 
                    pd.DataFrame(list(new_forecast["ens_mean"].shift(-6)), columns =["ens_mean_6"]).reset_index(drop = True)], 
                    axis=1)

  new_forecast_36 = new_forecast.drop(columns = ["ens_mean_-2", "ens_mean_-1", "ens_mean_1", "ens_mean_2"])
  new_forecast_36 = new_forecast_36[new_forecast_36.iloc[:,0] == 36].reset_index().iloc[:,1:]

  new_forecast_48 = new_forecast.drop(columns = ["ens_mean_-2", "ens_mean_-1", "ens_mean_1", "ens_mean_6"])
  new_forecast_48 = new_forecast_48[new_forecast_48.iloc[:,0] == 48].reset_index().iloc[:,1:]

  new_forecast_60 = new_forecast.drop(columns = ["ens_mean_-6", "ens_mean_-1", "ens_mean_1", "ens_mean_6"])
  new_forecast_60 = new_forecast_60[new_forecast_60.iloc[:,0] == 60].reset_index().iloc[:,1:]

  new_forecast_72 = new_forecast.drop(columns = ["ens_mean_-6", "ens_mean_-1", "ens_mean_6", "ens_mean_2"])
  new_forecast_72 = new_forecast_72[new_forecast_72.iloc[:,0] == 72].reset_index().iloc[:,1:]

  new_forecast_84 = new_forecast.drop(columns = ["ens_mean_-2", "ens_mean_-6", "ens_mean_6", "ens_mean_2"])
  new_forecast_84 = new_forecast_84[new_forecast_84.iloc[:,0] == 84].reset_index().iloc[:,1:]

  return new_forecast_36, new_forecast_48, new_forecast_60, new_forecast_72, new_forecast_84

#%%
#
#
# Temperature


df_temp = pd.read_excel("Temp_Input.xlsx")
df_temp = df_temp.iloc[:,1:]

X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test = Prepare_Other(df_temp)
new_forecast_36, new_forecast_48, new_forecast_60, new_forecast_72, new_forecast_84 = Prepare_new_Input('20220209', 2, add_month=False, wind =False )

QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

results_gb = np.concatenate([gb_quantile_t2(X_36_train.iloc[:,42:46], y_36_train, new_forecast_36.iloc[:,41:45], q) for q in QUANTILES]) 
result36 = ReshapeHour(results_gb)

results_gb = np.concatenate([gb_quantile_t1(X_48_train.iloc[:,42:46], y_48_train, new_forecast_48.iloc[:,41:45], q) for q in QUANTILES]) 
result48 = ReshapeHour(results_gb)

results_gb = np.concatenate([gb_quantile_t2(X_60_train.iloc[:,42:46], y_60_train, new_forecast_60.iloc[:,41:45], q) for q in QUANTILES]) 
result60 = ReshapeHour(results_gb)

results_gb = np.concatenate([gb_quantile_t1(X_72_train.iloc[:,42:46], y_72_train, new_forecast_72.iloc[:,41:45], q) for q in QUANTILES]) 
result72 = ReshapeHour(results_gb)

results_gb = np.concatenate([gb_quantile_t2(X_84_train.iloc[:,42:46], y_84_train, new_forecast_84.iloc[:,41:45], q) for q in QUANTILES]) 
result84 = ReshapeHour(results_gb)

result_temp = pd.concat([result36, result48, result60, result72, result84])

print(result_temp)
# %%
#
#
# Wind

df_wind = pd.read_excel("Wind_Input.xlsx")
df_wind = df_wind.iloc[:,1:]

X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test = Prepare_Other(df_wind)
new_forecast_36, new_forecast_48, new_forecast_60, new_forecast_72, new_forecast_84 = Prepare_new_Input('20220209', 2, add_month=True, wind =True )

col_1_40                          = X_36_train.columns[2:42]
col_mean_var                      = X_36_train.columns[42:44]
col_month                         = X_36_train.columns[46:57] 
col_1_40_mean_var_month           = [*col_1_40,*col_mean_var, *col_month]

QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

results_gb = np.concatenate([gb_quantile(X_36_train[col_1_40_mean_var_month], y_36_train, new_forecast_36.iloc[:,1:54], q) for q in QUANTILES]) 
result36 = ReshapeHour(results_gb)

results_gb = np.concatenate([gb_quantile(X_48_train[col_1_40_mean_var_month], y_48_train, new_forecast_48.iloc[:,1:54], q) for q in QUANTILES]) 
result48 = ReshapeHour(results_gb)

results_gb = np.concatenate([gb_quantile(X_60_train[col_1_40_mean_var_month], y_60_train, new_forecast_60.iloc[:,1:54], q) for q in QUANTILES]) 
result60 = ReshapeHour(results_gb)

results_gb = np.concatenate([gb_quantile(X_72_train[col_1_40_mean_var_month], y_72_train, new_forecast_72.iloc[:,1:54], q) for q in QUANTILES]) 
result72 = ReshapeHour(results_gb)

results_gb = np.concatenate([gb_quantile(X_84_train[col_1_40_mean_var_month], y_84_train, new_forecast_84.iloc[:,1:54], q) for q in QUANTILES]) 
result84 = ReshapeHour(results_gb)

result_wind = pd.concat([result36, result48, result60, result72, result84])

print(result_wind)

# %%
