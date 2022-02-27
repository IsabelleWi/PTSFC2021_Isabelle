
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

from sklearn import ensemble
import statsmodels.api as sm
from scipy.stats import norm

from Seq_to_Seq_Functions import *
from Preprocessing_Evaluation_Functions import *
from Models_Functions import *

def ols_quantile(m, X, var, q):

    mean_pred = m.predict(X)
    se = np.sqrt(var.T)

    return mean_pred + norm.ppf(q) * se

    
def OLS_with_Quantiles(X_train, y_train, X_test, X_var_test):

  ols = sm.OLS(np.asarray(X_train, dtype=float), 
               np.asarray(y_train, dtype=float)).fit()

  QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

  ols_results_ret= np.stack(
    [ols_quantile(ols, np.asarray(X_test), X_var_test, q) 
    for q in QUANTILES], 
    axis=1) 
  
  return pd.DataFrame(ols_results_ret[0,:,:])

def Train_OLS(X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test):
    
    o1 = OLS_with_Quantiles(X_36_train.iloc[:,42:43], y_36_train, X_36_test.iloc[:,42:43], X_36_test.iloc[:,43:44])
    o2 = OLS_with_Quantiles(X_48_train.iloc[:,42:43], y_48_train, X_48_test.iloc[:,42:43], X_48_test.iloc[:,43:44])
    o3 = OLS_with_Quantiles(X_60_train.iloc[:,42:43], y_60_train, X_60_test.iloc[:,42:43], X_60_test.iloc[:,43:44])
    o4 = OLS_with_Quantiles(X_72_train.iloc[:,42:43], y_72_train, X_72_test.iloc[:,42:43], X_72_test.iloc[:,43:44])
    o5 = OLS_with_Quantiles(X_84_train.iloc[:,42:43], y_84_train, X_84_test.iloc[:,42:43], X_84_test.iloc[:,43:44])

    R_O_1 =Backtesting_per_Hour(o1.T, y_36_test, '1')
    R_O_2 =Backtesting_per_Hour(o2.T, y_48_test, '2')
    R_O_3 =Backtesting_per_Hour(o3.T, y_60_test, '3')
    R_O_4 =Backtesting_per_Hour(o4.T, y_72_test, '4')
    R_O_5 =Backtesting_per_Hour(o5.T, y_84_test, '5')
    df_o = pd.concat([R_O_1, R_O_2, R_O_3, R_O_4, R_O_5], axis = 1)

    df_o['Average'] = df_o.mean(axis=1)

    return df_o


def Benchmark(X_36_test, y_36_test, X_48_test,y_48_test, X_60_test,y_60_test, X_72_test, y_72_test,X_84_test, y_84_test):
    
    result = {}
    counter = 0

    for i in [X_36_test, X_48_test, X_60_test, X_72_test, X_84_test]:

        df = i
        results_975 = df.iloc[:,2:42].quantile(.975, axis=1)
        results_75 = df.iloc[:,2:42].quantile(.75, axis=1)
        results_5 = df.iloc[:,2:42].quantile(.5, axis=1)
        results_25 = df.iloc[:,2:42].quantile(.25, axis=1)
        results_025 = df.iloc[:,2:42].quantile(.025, axis=1)


        result[counter] =pd.concat([results_025, results_25, results_5, results_75, results_975], axis = 1)
        counter += 1

    results_s36 = Backtesting_per_Hour(result[0], y_36_test, '36')
    results_s48 = Backtesting_per_Hour(result[1], y_48_test, '48')
    results_s60 = Backtesting_per_Hour(result[2], y_60_test, '60')
    results_s72 = Backtesting_per_Hour(result[3], y_72_test, '72')
    results_s84 = Backtesting_per_Hour(result[4], y_84_test, '84')

    result_bench = pd.concat([pd.DataFrame(results_s36.values, columns = ['36'], index = results_s36.index), 
                                    pd.DataFrame(results_s48.values, columns = ['48'], index = results_s36.index), 
                                    pd.DataFrame(results_s60.values, columns = ['60'], index = results_s36.index), 
                                    pd.DataFrame(results_s72.values, columns = ['72'], index = results_s36.index), 
                                    pd.DataFrame(results_s84.values, columns = ['84'], index = results_s36.index)], axis = 1)

    result_bench['Average'] = result_bench.mean(axis=1)

    return result_bench

def gb_quantile(X_train, train_labels, X, q):
    N_ESTIMATORS=75
    gbf = ensemble.GradientBoostingRegressor(loss='quantile', alpha=q,
                                             n_estimators=N_ESTIMATORS,
                                             max_depth=6,
                                             learning_rate=0.06, min_samples_leaf=23,
                                             min_samples_split=23, random_state = 42)
    gbf.fit(X_train, train_labels)
    return gbf.predict(X)

def gb_quantile_t1(X_train, train_labels, X, q):
    N_ESTIMATORS=125
    gbf = ensemble.GradientBoostingRegressor(loss='quantile', alpha=q,
                                             n_estimators=N_ESTIMATORS,
                                             max_depth=2,
                                             learning_rate=0.08, min_samples_leaf=8,
                                             min_samples_split=10, random_state = 42)
    gbf.fit(X_train, train_labels)
    return gbf.predict(X)

def gb_quantile_t2(X_train, train_labels, X, q):
    N_ESTIMATORS=125
    gbf = ensemble.GradientBoostingRegressor(loss='quantile', alpha=q,
                                             n_estimators=N_ESTIMATORS,
                                             max_depth=6,
                                             learning_rate=0.08, min_samples_leaf=24,
                                             min_samples_split=25, random_state = 42)
    gbf.fit(X_train, train_labels)
    return gbf.predict(X)


def TrainGB(X_train, X_test, y_train, y_test, hour):

    QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

    results_gb = np.concatenate(
        [gb_quantile(X_train, y_train, X_test, q) for q in QUANTILES]) 
    
    result = ReshapeHour(results_gb)

    results = Backtesting_per_Hour(result, y_test, hour)

    return results

def TrainGB_t1(X_train, X_test, y_train, y_test, hour):

    QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

    results_gb = np.concatenate(
        [gb_quantile_t1(X_train, y_train, X_test, q) for q in QUANTILES]) 
    
    result = ReshapeHour(results_gb)

    results = Backtesting_per_Hour(result, y_test, hour)

    return results

def TrainGB_t2(X_train, X_test, y_train, y_test, hour):

    QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

    results_gb = np.concatenate(
        [gb_quantile_t2(X_train, y_train, X_test, q) for q in QUANTILES]) 
    
    result = ReshapeHour(results_gb)

    results = Backtesting_per_Hour(result, y_test, hour)

    return results

def Train_GB_Temp(X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test, name):
   
    col_1_40                          = X_36_train.columns[2:42]
    col_mean_var                      = X_36_train.columns[42:44]
    col_6_6                           = X_36_train.columns[44:46]
    col_1_40_mean_var_6_6             = [*col_1_40, *col_mean_var, *col_6_6]
    r_col_1_40_mean_var_6_6           = TrainGB_t2(X_36_train[col_1_40_mean_var_6_6], X_36_test[col_1_40_mean_var_6_6], y_36_train, y_36_test, '36')
    results_cat_36                    = r_col_1_40_mean_var_6_6

    col_1_40                          = X_48_train.columns[2:42]
    col_mean_var                      = X_48_train.columns[42:44]
    col_6_6                           = X_48_train.columns[44:46]
    col_1_40_mean_var_6_6             = [*col_1_40, *col_mean_var, *col_6_6]
    
    r_col_1_40_mean_var_6_6           = TrainGB_t1(X_48_train[col_1_40_mean_var_6_6], X_48_test[col_1_40_mean_var_6_6], y_48_train, y_48_test, '48')
    
    results_cat_48                    = r_col_1_40_mean_var_6_6

    col_1_40                          = X_60_train.columns[2:42]
    col_mean_var                      = X_60_train.columns[42:44]
    col_6_6                           = X_60_train.columns[44:46]
    col_1_40_mean_var_6_6             = [*col_1_40, *col_mean_var, *col_6_6]
    
    r_col_1_40_mean_var_6_6           = TrainGB_t2(X_60_train[col_1_40_mean_var_6_6], X_60_test[col_1_40_mean_var_6_6], y_60_train, y_60_test, '60')
   
    results_cat_60                    = r_col_1_40_mean_var_6_6

    col_1_40                          = X_72_train.columns[2:42]
    col_mean_var                      = X_72_train.columns[42:44]
    col_6_6                           = X_72_train.columns[44:46]
    col_1_40_mean_var_6_6             = [*col_1_40, *col_mean_var, *col_6_6]
    
    r_col_1_40_mean_var_6_6           = TrainGB_t1(X_72_train[col_1_40_mean_var_6_6], X_72_test[col_1_40_mean_var_6_6], y_72_train, y_72_test, '72')
    
    results_cat_72                   = r_col_1_40_mean_var_6_6

    col_1_40                          = X_84_train.columns[2:42]
    col_mean_var                      = X_84_train.columns[42:44]
    col_6_6                           = X_84_train.columns[44:46]
    col_1_40_mean_var_6_6             = [*col_1_40, *col_mean_var, *col_6_6]
    
    r_col_1_40_mean_var_6_6           = TrainGB_t2(X_84_train[col_1_40_mean_var_6_6], X_84_test[col_1_40_mean_var_6_6], y_84_train, y_84_test, '84')
    results_cat_84                    = r_col_1_40_mean_var_6_6

    r0 = pd.concat([results_cat_36.iloc[:,0], results_cat_48.iloc[:,0], results_cat_60.iloc[:,0], results_cat_72.iloc[:,0], results_cat_84.iloc[:,0]], axis = 1)
    r0['Average'] = r0.mean(axis=1)

    return r0

def Train_GB_Wind(X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test, name):
    
    col_1_40                          = X_36_train.columns[2:42]
    col_mean_var                      = X_36_train.columns[42:44]
    col_month                         = X_36_train.columns[46:57] #leave out one to avoid multicollinearity
    col_1_40_mean_var_month           = [*col_1_40,*col_mean_var, *col_month]

    r_col_1_40_mean_var_month         = TrainGB(X_36_train[col_1_40_mean_var_month], X_36_test[col_1_40_mean_var_month], y_36_train, y_36_test, '36')
    results_cat_36                    = r_col_1_40_mean_var_month

    r_col_1_40_mean_var_month         = TrainGB(X_48_train[col_1_40_mean_var_month], X_48_test[col_1_40_mean_var_month], y_48_train, y_48_test, '48')
    results_cat_48                    = r_col_1_40_mean_var_month

    r_col_1_40_mean_var_month         = TrainGB(X_60_train[col_1_40_mean_var_month], X_60_test[col_1_40_mean_var_month], y_60_train, y_60_test, '60')
    results_cat_60                   = r_col_1_40_mean_var_month
    
    r_col_1_40_mean_var_month         = TrainGB(X_72_train[col_1_40_mean_var_month], X_72_test[col_1_40_mean_var_month], y_72_train, y_72_test, '72')
    results_cat_72                   = r_col_1_40_mean_var_month

    r_col_1_40_mean_var_month         = TrainGB(X_84_train[col_1_40_mean_var_month], X_84_test[col_1_40_mean_var_month], y_84_train, y_84_test, '84')
    results_cat_84                   = r_col_1_40_mean_var_month

    r0 = pd.concat([results_cat_36.iloc[:,0], results_cat_48.iloc[:,0], results_cat_60.iloc[:,0], results_cat_72.iloc[:,0], results_cat_84.iloc[:,0]], axis = 1)
    r0['Average'] = r0.mean(axis=1)

    return r0

def GB_Feature_Testing(X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test, name):
    col_1_40                          = X_36_train.columns[2:42]
    col_mean_var                      = X_36_train.columns[42:44]
    col_6_6                           = X_36_train.columns[44:46]
    col_month                         = X_36_train.columns[46:57] #leave out one to avoid multicollinearity
    col_1_40_mean_var                 = [*col_1_40,*col_mean_var]
    col_1_40_mean_var_6_6             = [*col_1_40, *col_mean_var, *col_6_6]
    col_1_40_mean_var_6_6_month       = [*col_1_40,*col_mean_var, *col_6_6, *col_month]
    col_mean_var_6_6                  = [*col_mean_var, *col_6_6]
    col_mean_var_6_6_month            = [*col_mean_var, *col_6_6, *col_month]
    col_mean_var_month                = [*col_mean_var, *col_month]
    col_1_40_6_6                      = [*col_1_40,*col_6_6]
    col_1_40_month                    = [*col_1_40, *col_month]
    col_6_6_month                     = [*col_6_6,*col_month]
    col_1_40_mean_var_month           = [*col_1_40,*col_mean_var, *col_month]


    r_col_1_40                        = TrainGB(X_36_train[col_1_40], X_36_test[col_1_40], y_36_train, y_36_test, '36')
    r_col_mean_var                    = TrainGB(X_36_train[col_mean_var], X_36_test[col_mean_var], y_36_train, y_36_test, '36')
    r_col_6_6                         = TrainGB(X_36_train[col_6_6], X_36_test[col_6_6], y_36_train, y_36_test, '36')
    r_col_month                       = TrainGB(X_36_train[col_month], X_36_test[col_month], y_36_train, y_36_test, '36')
    r_col_1_40_mean_var               = TrainGB(X_36_train[col_1_40_mean_var], X_36_test[col_1_40_mean_var], y_36_train, y_36_test, '36')
    r_col_1_40_mean_var_6_6           = TrainGB(X_36_train[col_1_40_mean_var_6_6], X_36_test[col_1_40_mean_var_6_6], y_36_train, y_36_test, '36')
    r_col_1_40_mean_var_6_6_month     = TrainGB(X_36_train[col_1_40_mean_var_6_6_month], X_36_test[col_1_40_mean_var_6_6_month], y_36_train, y_36_test, '36')
    r_col_mean_var_6_6                = TrainGB(X_36_train[col_mean_var_6_6], X_36_test[col_mean_var_6_6], y_36_train, y_36_test, '36')
    r_col_mean_var_6_6_month          = TrainGB(X_36_train[col_mean_var_6_6_month], X_36_test[col_mean_var_6_6_month], y_36_train, y_36_test, '36')
    r_col_mean_var_month              = TrainGB(X_36_train[col_mean_var_month], X_36_test[col_mean_var_month], y_36_train, y_36_test, '36')
    r_col_1_40_6_6                    = TrainGB(X_36_train[col_1_40_6_6], X_36_test[col_1_40_6_6], y_36_train, y_36_test, '36')
    r_col_1_40_month                  = TrainGB(X_36_train[col_1_40_month], X_36_test[col_1_40_month], y_36_train, y_36_test, '36')
    r_col_6_6_month                   = TrainGB(X_36_train[col_6_6_month], X_36_test[col_6_6_month], y_36_train, y_36_test, '36')
    r_col_1_40_mean_var_month         = TrainGB(X_36_train[col_1_40_mean_var_month], X_36_test[col_1_40_mean_var_month], y_36_train, y_36_test, '36')


    results_cat_36                    = pd.concat([r_col_1_40, r_col_mean_var, r_col_6_6, r_col_month, r_col_1_40_mean_var, r_col_1_40_6_6, r_col_1_40_month,r_col_mean_var_6_6, r_col_mean_var_month, r_col_6_6_month,r_col_1_40_mean_var_6_6, r_col_1_40_mean_var_month, r_col_mean_var_6_6_month, r_col_1_40_mean_var_6_6_month], axis = 1)

    col_1_40                          = X_48_train.columns[2:42]
    col_mean_var                      = X_48_train.columns[42:44]
    col_6_6                           = X_48_train.columns[44:46]
    col_month                         = X_48_train.columns[46:57] #leave out one to avoid multicollinearity
    col_1_40_mean_var                 = [*col_1_40,*col_mean_var]
    col_1_40_mean_var_6_6             = [*col_1_40, *col_mean_var, *col_6_6]
    col_1_40_mean_var_6_6_month       = [*col_1_40,*col_mean_var, *col_6_6, *col_month]
    col_mean_var_6_6                  = [*col_mean_var, *col_6_6]
    col_mean_var_6_6_month            = [*col_mean_var, *col_6_6, *col_month]
    col_mean_var_month                = [*col_mean_var, *col_month]
    col_1_40_6_6                      = [*col_1_40,*col_6_6]
    col_1_40_month                    = [*col_1_40, *col_month]
    col_6_6_month                     = [*col_6_6,*col_month]
    col_1_40_mean_var_month           = [*col_1_40,*col_mean_var, *col_month]


    r_col_1_40                        = TrainGB(X_48_train[col_1_40], X_48_test[col_1_40], y_48_train, y_48_test, '48')
    r_col_mean_var                    = TrainGB(X_48_train[col_mean_var], X_48_test[col_mean_var], y_48_train, y_48_test, '48')
    r_col_6_6                         = TrainGB(X_48_train[col_6_6], X_48_test[col_6_6], y_48_train, y_48_test, '48')
    r_col_month                       = TrainGB(X_48_train[col_month], X_48_test[col_month], y_48_train, y_48_test, '48')
    r_col_1_40_mean_var               = TrainGB(X_48_train[col_1_40_mean_var], X_48_test[col_1_40_mean_var], y_48_train, y_48_test, '48')
    r_col_1_40_mean_var_6_6           = TrainGB(X_48_train[col_1_40_mean_var_6_6], X_48_test[col_1_40_mean_var_6_6], y_48_train, y_48_test, '48')
    r_col_1_40_mean_var_6_6_month     = TrainGB(X_48_train[col_1_40_mean_var_6_6_month], X_48_test[col_1_40_mean_var_6_6_month], y_48_train, y_48_test, '48')
    r_col_mean_var_6_6                = TrainGB(X_48_train[col_mean_var_6_6], X_48_test[col_mean_var_6_6], y_48_train, y_48_test, '48')
    r_col_mean_var_6_6_month          = TrainGB(X_48_train[col_mean_var_6_6_month], X_48_test[col_mean_var_6_6_month], y_48_train, y_48_test, '48')
    r_col_mean_var_month              = TrainGB(X_48_train[col_mean_var_month], X_48_test[col_mean_var_month], y_48_train, y_48_test, '48')
    r_col_1_40_6_6                    = TrainGB(X_48_train[col_1_40_6_6], X_48_test[col_1_40_6_6], y_48_train, y_48_test, '48')
    r_col_1_40_month                  = TrainGB(X_48_train[col_1_40_month], X_48_test[col_1_40_month], y_48_train, y_48_test, '48')
    r_col_6_6_month                   = TrainGB(X_48_train[col_6_6_month], X_48_test[col_6_6_month], y_48_train, y_48_test, '48')
    r_col_1_40_mean_var_month         = TrainGB(X_48_train[col_1_40_mean_var_month], X_48_test[col_1_40_mean_var_month], y_48_train, y_48_test, '48')


    results_cat_48                    = pd.concat([r_col_1_40, r_col_mean_var, r_col_6_6, r_col_month, r_col_1_40_mean_var, r_col_1_40_6_6, r_col_1_40_month,r_col_mean_var_6_6, r_col_mean_var_month, r_col_6_6_month,r_col_1_40_mean_var_6_6, r_col_1_40_mean_var_month, r_col_mean_var_6_6_month, r_col_1_40_mean_var_6_6_month], axis = 1)

    col_1_40                          = X_60_train.columns[2:42]
    col_mean_var                      = X_60_train.columns[42:44]
    col_6_6                           = X_60_train.columns[44:46]
    col_month                         = X_60_train.columns[46:57] #leave out one to avoid multicollinearity
    col_1_40_mean_var                 = [*col_1_40,*col_mean_var]
    col_1_40_mean_var_6_6             = [*col_1_40, *col_mean_var, *col_6_6]
    col_1_40_mean_var_6_6_month       = [*col_1_40,*col_mean_var, *col_6_6, *col_month]
    col_mean_var_6_6                  = [*col_mean_var, *col_6_6]
    col_mean_var_6_6_month            = [*col_mean_var, *col_6_6, *col_month]
    col_mean_var_month                = [*col_mean_var, *col_month]
    col_1_40_6_6                      = [*col_1_40,*col_6_6]
    col_1_40_month                    = [*col_1_40, *col_month]
    col_6_6_month                     = [*col_6_6,*col_month]
    col_1_40_mean_var_month           = [*col_1_40,*col_mean_var, *col_month]


    r_col_1_40                        = TrainGB(X_60_train[col_1_40], X_60_test[col_1_40], y_60_train, y_60_test, '60')
    r_col_mean_var                    = TrainGB(X_60_train[col_mean_var], X_60_test[col_mean_var], y_60_train, y_60_test, '60')
    r_col_6_6                         = TrainGB(X_60_train[col_6_6], X_60_test[col_6_6], y_60_train, y_60_test, '60')
    r_col_month                       = TrainGB(X_60_train[col_month], X_60_test[col_month], y_60_train, y_60_test, '60')

    r_col_1_40_mean_var               = TrainGB(X_60_train[col_1_40_mean_var], X_60_test[col_1_40_mean_var], y_60_train, y_60_test, '60')
    r_col_1_40_mean_var_6_6           = TrainGB(X_60_train[col_1_40_mean_var_6_6], X_60_test[col_1_40_mean_var_6_6], y_60_train, y_60_test, '60')
    r_col_1_40_mean_var_6_6_month     = TrainGB(X_60_train[col_1_40_mean_var_6_6_month], X_60_test[col_1_40_mean_var_6_6_month], y_60_train, y_60_test, '60')
    r_col_mean_var_6_6                = TrainGB(X_60_train[col_mean_var_6_6], X_60_test[col_mean_var_6_6], y_60_train, y_60_test, '60')
    r_col_mean_var_6_6_month          = TrainGB(X_60_train[col_mean_var_6_6_month], X_60_test[col_mean_var_6_6_month], y_60_train, y_60_test, '60')
    r_col_mean_var_month              = TrainGB(X_60_train[col_mean_var_month], X_60_test[col_mean_var_month], y_60_train, y_60_test, '60')

    r_col_1_40_6_6                    = TrainGB(X_60_train[col_1_40_6_6], X_60_test[col_1_40_6_6], y_60_train, y_60_test, '60')
    r_col_1_40_month                  = TrainGB(X_60_train[col_1_40_month], X_60_test[col_1_40_month], y_60_train, y_60_test, '60')
    r_col_6_6_month                   = TrainGB(X_60_train[col_6_6_month], X_60_test[col_6_6_month], y_60_train, y_60_test, '60')
    r_col_1_40_mean_var_month         = TrainGB(X_60_train[col_1_40_mean_var_month], X_60_test[col_1_40_mean_var_month], y_60_train, y_60_test, '60')


    results_cat_60                   = pd.concat([r_col_1_40, r_col_mean_var, r_col_6_6, r_col_month, r_col_1_40_mean_var, r_col_1_40_6_6, r_col_1_40_month,r_col_mean_var_6_6, r_col_mean_var_month, r_col_6_6_month,r_col_1_40_mean_var_6_6, r_col_1_40_mean_var_month, r_col_mean_var_6_6_month, r_col_1_40_mean_var_6_6_month], axis = 1)

    col_1_40                          = X_72_train.columns[2:42]
    col_mean_var                      = X_72_train.columns[42:44]
    col_6_6                           = X_72_train.columns[44:46]
    col_month                         = X_72_train.columns[46:57] #leave out one to avoid multicollinearity
    col_1_40_mean_var                 = [*col_1_40,*col_mean_var]
    col_1_40_mean_var_6_6             = [*col_1_40, *col_mean_var, *col_6_6]
    col_1_40_mean_var_6_6_month       = [*col_1_40,*col_mean_var, *col_6_6, *col_month]
    col_mean_var_6_6                  = [*col_mean_var, *col_6_6]
    col_mean_var_6_6_month            = [*col_mean_var, *col_6_6, *col_month]
    col_mean_var_month                = [*col_mean_var, *col_month]
    col_1_40_6_6                      = [*col_1_40,*col_6_6]
    col_1_40_month                    = [*col_1_40, *col_month]
    col_6_6_month                     = [*col_6_6,*col_month]
    col_1_40_mean_var_month           = [*col_1_40,*col_mean_var, *col_month]


    r_col_1_40                        = TrainGB(X_72_train[col_1_40], X_72_test[col_1_40], y_72_train, y_72_test, '72')
    r_col_mean_var                    = TrainGB(X_72_train[col_mean_var], X_72_test[col_mean_var], y_72_train, y_72_test, '72')
    r_col_6_6                         = TrainGB(X_72_train[col_6_6], X_72_test[col_6_6], y_72_train, y_72_test, '72')
    r_col_month                       = TrainGB(X_72_train[col_month], X_72_test[col_month], y_72_train, y_72_test, '72')

    r_col_1_40_mean_var               = TrainGB(X_72_train[col_1_40_mean_var], X_72_test[col_1_40_mean_var], y_72_train, y_72_test, '72')
    r_col_1_40_mean_var_6_6           = TrainGB(X_72_train[col_1_40_mean_var_6_6], X_72_test[col_1_40_mean_var_6_6], y_72_train, y_72_test, '72')
    r_col_1_40_mean_var_6_6_month     = TrainGB(X_72_train[col_1_40_mean_var_6_6_month], X_72_test[col_1_40_mean_var_6_6_month], y_72_train, y_72_test, '72')
    r_col_mean_var_6_6                = TrainGB(X_72_train[col_mean_var_6_6], X_72_test[col_mean_var_6_6], y_72_train, y_72_test, '72')
    r_col_mean_var_6_6_month          = TrainGB(X_72_train[col_mean_var_6_6_month], X_72_test[col_mean_var_6_6_month], y_72_train, y_72_test, '72')
    r_col_mean_var_month              = TrainGB(X_72_train[col_mean_var_month], X_72_test[col_mean_var_month], y_72_train, y_72_test, '72')

    r_col_1_40_6_6                    = TrainGB(X_72_train[col_1_40_6_6], X_72_test[col_1_40_6_6], y_72_train, y_72_test, '72')
    r_col_1_40_month                  = TrainGB(X_72_train[col_1_40_month], X_72_test[col_1_40_month], y_72_train, y_72_test, '72')
    r_col_6_6_month                   = TrainGB(X_72_train[col_6_6_month], X_72_test[col_6_6_month], y_72_train, y_72_test, '72')
    r_col_1_40_mean_var_month         = TrainGB(X_72_train[col_1_40_mean_var_month], X_72_test[col_1_40_mean_var_month], y_72_train, y_72_test, '72')


    results_cat_72                   = pd.concat([r_col_1_40, r_col_mean_var, r_col_6_6, r_col_month, r_col_1_40_mean_var, r_col_1_40_6_6, r_col_1_40_month,r_col_mean_var_6_6, r_col_mean_var_month, r_col_6_6_month,r_col_1_40_mean_var_6_6, r_col_1_40_mean_var_month, r_col_mean_var_6_6_month, r_col_1_40_mean_var_6_6_month], axis = 1)

    col_1_40                          = X_84_train.columns[2:42]
    col_mean_var                      = X_84_train.columns[42:44]
    col_6_6                           = X_84_train.columns[44:46]
    col_month                         = X_84_train.columns[46:57] #leave out one to avoid multicollinearity
    col_1_40_mean_var                 = [*col_1_40,*col_mean_var]
    col_1_40_mean_var_6_6             = [*col_1_40, *col_mean_var, *col_6_6]
    col_1_40_mean_var_6_6_month       = [*col_1_40,*col_mean_var, *col_6_6, *col_month]
    col_mean_var_6_6                  = [*col_mean_var, *col_6_6]
    col_mean_var_6_6_month            = [*col_mean_var, *col_6_6, *col_month]
    col_mean_var_month                = [*col_mean_var, *col_month]
    col_1_40_6_6                      = [*col_1_40,*col_6_6]
    col_1_40_month                    = [*col_1_40, *col_month]
    col_6_6_month                     = [*col_6_6,*col_month]
    col_1_40_mean_var_month           = [*col_1_40,*col_mean_var, *col_month]


    r_col_1_40                        = TrainGB(X_84_train[col_1_40], X_84_test[col_1_40], y_84_train, y_84_test, '84')
    r_col_mean_var                    = TrainGB(X_84_train[col_mean_var], X_84_test[col_mean_var], y_84_train, y_84_test, '84')
    r_col_6_6                         = TrainGB(X_84_train[col_6_6], X_84_test[col_6_6], y_84_train, y_84_test, '84')
    r_col_month                       = TrainGB(X_84_train[col_month], X_84_test[col_month], y_84_train, y_84_test, '84')

    r_col_1_40_mean_var               = TrainGB(X_84_train[col_1_40_mean_var], X_84_test[col_1_40_mean_var], y_84_train, y_84_test, '84')
    r_col_1_40_mean_var_6_6           = TrainGB(X_84_train[col_1_40_mean_var_6_6], X_84_test[col_1_40_mean_var_6_6], y_84_train, y_84_test, '84')
    r_col_1_40_mean_var_6_6_month     = TrainGB(X_84_train[col_1_40_mean_var_6_6_month], X_84_test[col_1_40_mean_var_6_6_month], y_84_train, y_84_test, '84')
    r_col_mean_var_6_6                = TrainGB(X_84_train[col_mean_var_6_6], X_84_test[col_mean_var_6_6], y_84_train, y_84_test, '84')
    r_col_mean_var_6_6_month          = TrainGB(X_84_train[col_mean_var_6_6_month], X_84_test[col_mean_var_6_6_month], y_84_train, y_84_test, '84')
    r_col_mean_var_month              = TrainGB(X_84_train[col_mean_var_month], X_84_test[col_mean_var_month], y_84_train, y_84_test, '84')

    r_col_1_40_6_6                    = TrainGB(X_84_train[col_1_40_6_6], X_84_test[col_1_40_6_6], y_84_train, y_84_test, '84')
    r_col_1_40_month                  = TrainGB(X_84_train[col_1_40_month], X_84_test[col_1_40_month], y_84_train, y_84_test, '84')
    r_col_6_6_month                   = TrainGB(X_84_train[col_6_6_month], X_84_test[col_6_6_month], y_84_train, y_84_test, '84')
    r_col_1_40_mean_var_month         = TrainGB(X_84_train[col_1_40_mean_var_month], X_84_test[col_1_40_mean_var_month], y_84_train, y_84_test, '84')


    results_cat_84                   = pd.concat([r_col_1_40, r_col_mean_var, r_col_6_6, r_col_month, r_col_1_40_mean_var, r_col_1_40_6_6, r_col_1_40_month,r_col_mean_var_6_6, r_col_mean_var_month, r_col_6_6_month,r_col_1_40_mean_var_6_6, r_col_1_40_mean_var_month, r_col_mean_var_6_6_month, r_col_1_40_mean_var_6_6_month], axis = 1)

    r0 = pd.concat([results_cat_36.iloc[:,0], results_cat_48.iloc[:,0], results_cat_60.iloc[:,0], results_cat_72.iloc[:,0], results_cat_84.iloc[:,0]], axis = 1)
    r1 = pd.concat([results_cat_36.iloc[:,1], results_cat_48.iloc[:,1], results_cat_60.iloc[:,1], results_cat_72.iloc[:,1], results_cat_84.iloc[:,1]], axis = 1)
    r2 = pd.concat([results_cat_36.iloc[:,2], results_cat_48.iloc[:,2], results_cat_60.iloc[:,2], results_cat_72.iloc[:,2], results_cat_84.iloc[:,2]], axis = 1)
    r3 = pd.concat([results_cat_36.iloc[:,3], results_cat_48.iloc[:,3], results_cat_60.iloc[:,3], results_cat_72.iloc[:,3], results_cat_84.iloc[:,3]], axis = 1)
    r4 = pd.concat([results_cat_36.iloc[:,4], results_cat_48.iloc[:,4], results_cat_60.iloc[:,4], results_cat_72.iloc[:,4], results_cat_84.iloc[:,4]], axis = 1)
    r5 = pd.concat([results_cat_36.iloc[:,5], results_cat_48.iloc[:,5], results_cat_60.iloc[:,5], results_cat_72.iloc[:,5], results_cat_84.iloc[:,5]], axis = 1)
    r6 = pd.concat([results_cat_36.iloc[:,6], results_cat_48.iloc[:,6], results_cat_60.iloc[:,6], results_cat_72.iloc[:,6], results_cat_84.iloc[:,6]], axis = 1)
    r7 = pd.concat([results_cat_36.iloc[:,7], results_cat_48.iloc[:,7], results_cat_60.iloc[:,7], results_cat_72.iloc[:,7], results_cat_84.iloc[:,7]], axis = 1)
    r8 = pd.concat([results_cat_36.iloc[:,8], results_cat_48.iloc[:,8], results_cat_60.iloc[:,8], results_cat_72.iloc[:,8], results_cat_84.iloc[:,8]], axis = 1)
    r9 = pd.concat([results_cat_36.iloc[:,9], results_cat_48.iloc[:,9], results_cat_60.iloc[:,9], results_cat_72.iloc[:,9], results_cat_84.iloc[:,9]], axis = 1)
    r10 = pd.concat([results_cat_36.iloc[:,10], results_cat_48.iloc[:,10], results_cat_60.iloc[:,10], results_cat_72.iloc[:,10], results_cat_84.iloc[:,10]], axis = 1)
    r11 = pd.concat([results_cat_36.iloc[:,11], results_cat_48.iloc[:,11], results_cat_60.iloc[:,11], results_cat_72.iloc[:,11], results_cat_84.iloc[:,11]], axis = 1)
    r12 = pd.concat([results_cat_36.iloc[:,12], results_cat_48.iloc[:,12], results_cat_60.iloc[:,12], results_cat_72.iloc[:,12], results_cat_84.iloc[:,12]], axis = 1)
    r13 = pd.concat([results_cat_36.iloc[:,13], results_cat_48.iloc[:,13], results_cat_60.iloc[:,13], results_cat_72.iloc[:,13], results_cat_84.iloc[:,13]], axis = 1)

    r0['Average'] = r0.mean(axis=1)
    r1['Average'] = r1.mean(axis=1)
    r2['Average'] = r2.mean(axis=1)
    r3['Average'] = r3.mean(axis=1)
    r4['Average'] = r4.mean(axis=1)
    r5['Average'] = r5.mean(axis=1)
    r6['Average'] = r6.mean(axis=1)
    r7['Average'] = r7.mean(axis=1)
    r8['Average'] = r8.mean(axis=1)
    r9['Average'] = r9.mean(axis=1)
    r10['Average'] = r10.mean(axis=1)
    r11['Average'] = r11.mean(axis=1)
    r12['Average'] = r12.mean(axis=1)
    r13['Average'] = r13.mean(axis=1)

    all = pd.concat([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13], axis =1)
    all.to_excel('Feature Test' + name + '.xlsx')