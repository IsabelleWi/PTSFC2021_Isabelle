# Imports

import pandas as pd
import numpy as np

from scipy.stats import genhyperbolic
from scipy.special import kn
from scipy.optimize import minimize

from arch import arch_model
from arch.__future__ import reindexing


def var_func(x):
    return kn(1, x) / (kn(0,x) * x)

def diff(x,a):
    yt = var_func(x)
    return (yt - a )**2

def Inverse(x):
  res = minimize(diff, 1.0, args=(x), method='Nelder-Mead', tol=1e-6)
  y = res.x[0]
  return y


def TrainGarchGenhyp(train_y, mean, alpha = 0.35, test_size = 360):

  results_975 = []
  results_75 = []
  results_5 = []
  results_25 = []
  results_025 = []

  for i in range(test_size):
      train = train_y[:-(test_size-i)]
      model = arch_model(train,  p=2, o=0, q=2, vol = 'GARCH')
      model_fit = model.fit(disp='off')
      pred = model_fit.forecast(horizon=1)

      var = pred.residual_variance.values[-1,:][0]
      alpha = Inverse(var)

      results = genhyperbolic.rvs(0,alpha, 0,alpha * mean[i], size = 10000)

      results_975.append(pd.DataFrame(results).quantile(.975, axis=0))
      results_75.append(pd.DataFrame(results).quantile(.75, axis=0))
      results_5.append(pd.DataFrame(results).quantile(.5, axis=0))
      results_25.append(pd.DataFrame(results).quantile(.25, axis=0))
      results_025.append(pd.DataFrame(results).quantile(.025, axis=0))
      

  return pd.concat([pd.DataFrame(results_025).reset_index(drop=True), 
                    pd.DataFrame(results_25).reset_index(drop=True), 
                    pd.DataFrame(results_5).reset_index(drop=True), 
                    pd.DataFrame(results_75).reset_index(drop=True), 
                    pd.DataFrame(results_975).reset_index(drop=True)],
                    axis = 1)
                    

def TrainGarchNormal(train_y, mean, alpha = 0.35, beta = 0.05, test_size = 360):

  results_975 = []
  results_75 = []
  results_5 = []
  results_25 = []
  results_025 = []

  for i in range(test_size):

      train = train_y[:-(test_size-i)]
      model = arch_model(train,  p=2, o=0, q=2, vol = 'GARCH')
      model_fit = model.fit(disp='off')
      pred = model_fit.forecast(horizon=1)

      results = np.random.normal(alpha * mean[i], np.sqrt(pred.variance.values[-1,:][0]) + beta * abs(mean[i]), 10000)

      results_975.append(pd.DataFrame(results).quantile(.975, axis=0))
      results_75.append(pd.DataFrame(results).quantile(.75, axis=0))
      results_5.append(pd.DataFrame(results).quantile(.5, axis=0))
      results_25.append(pd.DataFrame(results).quantile(.25, axis=0))
      results_025.append(pd.DataFrame(results).quantile(.025, axis=0))
      

  return pd.concat([pd.DataFrame(results_025).reset_index(drop=True), 
                    pd.DataFrame(results_25).reset_index(drop=True), 
                    pd.DataFrame(results_5).reset_index(drop=True), 
                    pd.DataFrame(results_75).reset_index(drop=True), 
                    pd.DataFrame(results_975).reset_index(drop=True)],
                    axis = 1)