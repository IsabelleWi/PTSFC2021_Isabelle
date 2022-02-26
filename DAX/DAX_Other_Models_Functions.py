# Imports

import pandas as pd
import numpy as np

from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import ensemble


def ols_quantile(m, X, q):

    mean_pred = m.predict(X)
    se = np.sqrt(m.scale)

    return mean_pred + norm.ppf(q) * se

    
def OLS_with_Quantiles(X_train, y_train, X_test, y_test):

  ols = sm.OLS(np.asarray(X_train[['x-1']], dtype=float), 
               np.asarray(y_train, dtype=float)).fit()

  QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

  ols_results_ret= np.stack(
    [ols_quantile(ols, np.asarray(X_test['x-1']), q) 
    for q in QUANTILES], 
    axis=1) 
  
  return pd.DataFrame(ols_results_ret)


def QuantReg(X_train_1, y_train_1, X_test_1, y_test_1):

  QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

  df      = pd.DataFrame({'x': X_train_1['x-1'], 
                          'x2': X_train_1['x-2'], 
                          'x3': X_train_1['x-3'], 
                          'x4': X_train_1['x-4'], 
                          'x5': X_train_1['x-5'], 
                          'y': y_train_1.iloc[:,0]})
  
  df_pred = pd.DataFrame({'x': X_test_1['x-1'], 
                          'x2': X_test_1['x-2'], 
                          'x3': X_test_1['x-3'], 
                          'x4': X_test_1['x-4'], 
                          'x5': X_test_1['x-5']}) 

  results = {}

  for qt in QUANTILES:

    quantreg = smf.quantreg('y ~ x + x2 + x3 + x4 + x5', df)
    res = quantreg.fit(q = qt)
    results[qt] = res.predict(df_pred)
  
  return pd.DataFrame(np.stack([results[0.025],
                                results[0.25],
                                results[0.5],
                                results[0.75],
                                results[0.975]] , axis = 1))


def rf_quantile(m, X, q):

    rf_preds = []

    for estimator in m.estimators_:
        rf_preds.append(estimator.predict(X))
    rf_preds = np.array(rf_preds).transpose() 
    return np.percentile(rf_preds, q * 100, axis=1)


def RF_with_Quantiles(X_train_1, y_train_1, X_test_1, y_test_1):

  N_ESTIMATORS = 100
  QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

  rf = ensemble.RandomForestRegressor(n_estimators=N_ESTIMATORS, 
                                      min_samples_leaf=1, random_state=3, 
                                      verbose=True, 
                                      n_jobs=-1) 
  
  rf.fit(np.asarray(X_train_1, dtype=float), np.asarray(y_train_1, dtype=float).ravel())

  rf_results_ret1 = np.stack(
                    [rf_quantile(rf, np.asarray(X_test_1, dtype=float), q) 
                    for q in QUANTILES], 
                    axis =1)
  
  return pd.DataFrame(rf_results_ret1)



def gb_quantile(X_train, train_labels, X, q):

  N_ESTIMATORS = 100

  gbf = ensemble.GradientBoostingRegressor(loss='quantile', alpha=q,
                                            n_estimators=N_ESTIMATORS,
                                            max_depth=3,
                                            learning_rate=0.1, min_samples_leaf=9,
                                            min_samples_split=9)
  gbf.fit(X_train, train_labels.ravel())
  return gbf.predict(X)


def GB_Quantiles(X_train_1, y_train_1, X_test_1, y_test_1):

  QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

  gbt_results_ret1= np.stack([gb_quantile(np.asarray(X_train_1, dtype=float), 
                                          np.asarray(y_train_1, dtype=float),
                                          np.asarray(X_test_1, dtype=float), q) 
                                          for q in QUANTILES], 
                                          axis =1) 
  
  return pd.DataFrame(gbt_results_ret1)
