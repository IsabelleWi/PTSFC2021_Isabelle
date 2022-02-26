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

from Weather_NN_Functions import *
from Weather_Preprocessing_Evaluation_Functions import *
from Weather_Other_Models_Functions import *


df_temp = pd.read_excel("Temp_Input.xlsx")
df_temp = df_temp.iloc[:,1:]

X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test = Prepare_Other(df_temp)
#%%
result_bench = Benchmark(X_36_test, y_36_test, X_48_test,y_48_test, X_60_test,y_60_test, X_72_test, y_72_test,X_84_test, y_84_test)
result_ols = Train_OLS(X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test)
result_gb = Train_GB_Temp(X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test, 'Temp')

results_temp = pd.concat([result_bench, result_ols, result_gb], keys= ['Ens Quantiles', 'OLS', 'GB'], axis = 1)
results_temp.to_excel("Temp Other Models.xlsx")
# %% 
df_wind = pd.read_excel("Wind_Input.xlsx")
df_wind = df_wind.iloc[:,1:]

X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test = Prepare_Other(df_wind)
#%%
result_bench = Benchmark(X_36_test, y_36_test, X_48_test,y_48_test, X_60_test,y_60_test, X_72_test, y_72_test,X_84_test, y_84_test)
result_ols = Train_OLS(X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test)
result_gb = Train_GB_Wind(X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test, 'Wind')

results_temp = pd.concat([result_bench, result_ols, result_gb], keys= ['Ens Quantiles', 'OLS', 'GB'], axis = 1)
results_temp.to_excel("Wind Other Models.xlsx")


# %% 
GB_Feature_Testing(X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test, 'Wind')
# %% 

