#Imports
import numpy as np
import pandas as pd

import plotly_express as px
import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm
tqdm.pandas()

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable


def Average(lst):
    return sum(lst) / len(lst)

def calculate_QuantileScore(tau, forecast, y):

  """

  Input:
  tau: Quantile (e.g. 0.25)
  forecast: Predicted value
  y: Observed value

  Output:
  QScore: Quantile Score
  
  """

  if (forecast > y):
    QScore = 2 * (1-tau) * (forecast-y)
  
  elif (y >= forecast):
    QScore = 2 * tau * (y-forecast)

  
  return QScore

def AddMonthDummies(input):

  """

  Input:
  input: DataFrame

  Output:
  df : DataFrame with transformed column to Dummy

  """

  df = input.copy()
  df['month'] = df['obs_tm'].dt.month
  df = pd.get_dummies(df, columns=["month"])

  return df

def GetForecastingHours(input, hours):

  """
  Input:
  input: DataFrame containing data for all possible forecasting hours with column named 'fcst_hour'
  hours: List with relevant hours

  Output:
  output: Dict for all relevant hours
  """
  output = {}

  for h in hours:

    ishour = input['fcst_hour'] == h

    output[str(h)] = input[ishour].reset_index().iloc[:,1:]

  return output

def MergeForecastingWindows(wind_hour):

  wind_tot = pd.concat([wind_hour['36'], wind_hour['48'], wind_hour['60'], wind_hour['72'], wind_hour['84']], axis = 0)

  return wind_tot

def ReshapeResults(results):

  """
  Input:
  results: Dict with predictions for five forecasting windows (36h, 48h, 60h, 72h, 84h) containing DataFrame in shape k x 1

  Output:
  results: Reshaped in n x 5 DataFrame

  """
  res = {}
  HOURS = [36, 48, 60, 72, 84]

  for h in HOURS:
    res[str(h)] = pd.DataFrame(np.reshape( results[str(h)], (5, int (len(results[str(h)]) / 5)))).T

  return res

def ReshapeHour(results):

  """
  Input:
  results: DataFrame in shape k x 1

  Output:
  results: Reshaped in n x 5 DataFrame

  """
  res = pd.DataFrame(np.reshape( results, (5, int (len(results) / 5)))).T

  return res


def Coverage(y_true, y_upper, y_lower):

  """
  Input:
  y_true, y_upper, y_lower: Dataframes in the shape (n,1)

  Output:
  Float indicating Coverage Percentage

  """

  y_true = list(y_true.T)
  y_upper = list(y_upper.T)
  y_lower = list(y_lower.T)

  Counter = 0

  for i in range(0,len(y_true)):

    if (y_true[i] <= y_upper[i]) & (y_true[i] >= y_lower[i]):

      Counter += 1
  
  return Counter/len(y_true)

def Backtesting_per_Hour(y_pred, y_true, hour = 'Not given', print_statistics = True, plot_data = True):

  """ 
  Input: (for one forecasting hour!)
  y_pred: Pandas Dataframe n x 5 with n being the number of predicted datapoints with 5 Quantiles (2.5%, 25%, 50%, 75%, 97.5%)
  y_true: Pandas Dataframe containing observed Data n x 1
  hour: String indentifying forecasting hour

  Output:
  Score: Sum of Quantile Scores over all n datapoints
  """

  shape = y_pred.shape

  print('-------------------------------------------------------------')
  print('Hour: ' + str(hour))
  print('\nNumber of predicted datapoints :', shape[0])
  print()

  if plot_data == True:
    plt.figure(figsize=(25,5))
    plt.plot(y_true.reset_index(drop=True), 'k-')
    plt.plot(y_pred.reset_index(drop=True).iloc[:,0], 'r:')
    plt.plot(y_pred.reset_index(drop=True).iloc[:,1], 'r--')
    plt.plot(y_pred.reset_index(drop=True).iloc[:,2], 'r-')
    plt.plot(y_pred.reset_index(drop=True).iloc[:,3], 'r--')
    plt.plot(y_pred.reset_index(drop=True).iloc[:,4], 'r:')
    plt.show()

  Score = {}
  QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

  for i in range(0,5):

    Score[str(QUANTILES[i]) + " QScore Sum"] =  sum([calculate_QuantileScore(QUANTILES[i], y_pred.reset_index(drop=True).iloc[j,i], y_true.reset_index(drop=True).iloc[j]) for j in range(0,shape[0])])
    Score[str(QUANTILES[i]) + " QScore Avg"] = Score[str(QUANTILES[i]) + " QScore Sum"]/shape[0]

  Score['QScore Sum'] = sum(Score[str(QUANTILES[i]) + " QScore Avg"] for i in range(0,5))
  Score['QScore Avg'] = sum(Score[str(QUANTILES[i]) + " QScore Avg"] for i in range(0,5))/5
  Score['MAE'] = mean_absolute_error(y_true, y_pred.iloc[:,2])
  Score['Avg Width 95% Quantile'] = Average(y_pred.iloc[:,4] - y_pred.iloc[:,0])
  Score['Avg Width 50% Quantile'] = Average(y_pred.iloc[:,3] - y_pred.iloc[:,1])
  Score['Avg Coverage 95% Quantile'] = Coverage(y_true, y_pred.iloc[:,4], y_pred.iloc[:,0])
  Score['Avg Coverage 50% Quantile'] = Coverage(y_true, y_pred.iloc[:,3], y_pred.iloc[:,1])

  Scores = pd.DataFrame([Score['QScore Sum'], Score['QScore Avg'], Score[str(QUANTILES[0]) + " QScore Avg"], Score[str(QUANTILES[1]) + " QScore Avg"], Score[str(QUANTILES[2]) + " QScore Avg"], Score[str(QUANTILES[3]) + " QScore Avg"], Score[str(QUANTILES[4]) + " QScore Avg"], Score['MAE'], Score['Avg Width 95% Quantile'], Score['Avg Width 50% Quantile'], Score['Avg Coverage 95% Quantile'], Score['Avg Coverage 50% Quantile']], 
                        index = ['QScore Sum', 'QScore Avg', 'QScore 0.025 Avg', 'QScore 0.25 Avg', 'QScore 0.5 Avg', 'QScore 0.75 Avg', 'QScore 0.975 Avg', 'MAE', 'Avg Width 95% Quantile', 'Avg Width 50% Quantile', 'Avg Coverage 95% Quantile', 'Avg Coverage 50% Quantile'])

  if print_statistics == True:
    print()
    print(Scores)
    print()
    print('-------------------------------------------------------------')

  return Scores


def Backtesting(y_pred, y_true, hours):

  """ 
  Input: (for one forecasting hour!)
  y_pred: Dict with 5 Pandas Dataframe n x 5 with n being the number of predicted datapoints with 5 Quantiles (2.5%, 25%, 50%, 75%, 97.5%)
  y_true: Dict with 5 Pandas Dataframe containing observed Data n x 1
  hour: List indentifying forecasting hours

  Output:
  Score: Sum of Quantile Scores over all n datapoints and all hours
  """

  Score = sum([Backtesting_per_Hour(y_pred[str(i)], y_true[str(i)], str(i)) for i in hours])
  print()
  print()
  print('THE TOTAL SCORE IS ' + str(Score))


def Remove_Outlier_and_Split(df_input, 
                            plot_outliers   = True,
                            columns_X       = ["fcst_hour"],
                            columns_y       = "obs"):
  """

  Input:
  df_input: Dataframe, required columns: "fcst_hour", "obs"
  plot_outliers: Boolean indicating wether Values and Outliers should be visualised
  columns_X: List with String of X columns to keep
  columns_y: String with name of target variable column

  Output:
  X_train, X_test, y_train, y_test: Dataframes without outliers, splitted 90/10

  """

  df_3sigma=df_input.copy()
  df_3sigma["std_delivery_time"]=df_3sigma.groupby(["fcst_hour"])[columns_y].transform("std")
  df_3sigma["mean_delivery_time"]=df_3sigma.groupby(["fcst_hour"])[columns_y].transform("mean")

  df_3sigma["three_sigma_obs"]=3*df_3sigma["std_delivery_time"]
  df_3sigma["threshold_obs_upper_limit"]= df_3sigma["mean_delivery_time"] + df_3sigma["three_sigma_obs"]
  df_3sigma["threshold_obs_lower_limit"]= df_3sigma["mean_delivery_time"] - df_3sigma["three_sigma_obs"]

  df_3sigma["Is_outlier"]=np.where((df_3sigma.obs > df_3sigma.threshold_obs_upper_limit)| (df_3sigma.obs < df_3sigma.threshold_obs_lower_limit) , True, False)


  #e.g. df_3sigma (90/10)

  df_train = df_3sigma[:int(df_3sigma.shape[0]*0.9)] #drop outliers
  df_train_noOutliers = df_train[df_train.Is_outlier == False]
  df_test = df_3sigma[int(df_3sigma.shape[0]*0.9):] #leave outliers

  X_train = df_train_noOutliers[columns_X]
  X_train_np = X_train.to_numpy()

  y_train = df_train_noOutliers[columns_y]
  y_train_np = y_train.to_numpy()

  X_test=df_test[columns_X]
  y_test=df_test[columns_y]

  #Print dimensions
  print("Size of X_train:", X_train.shape)
  print("Size of X_test:", X_test.shape)
  print("Size of y_train:", y_train.shape)
  print("Size of y_test:", y_test.shape)

  return X_train, X_test, y_train, y_test


def Prepare_NN(data_weather):

    print('Shape: ',data_weather.shape)

    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(61)), columns =["ens_mean_-61"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(59)), columns =["ens_mean_-59"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(57)), columns =["ens_mean_-57"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(49)), columns =["ens_mean_-49"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(6)), columns =["ens_mean_-6"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(2)), columns =["ens_mean_-2"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(1)), columns =["ens_mean_-1"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(-1)), columns =["ens_mean_1"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(-2)), columns =["ens_mean_2"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(-6)), columns =["ens_mean_6"]).reset_index(drop = True)], axis=1)

    data_weather = data_weather.dropna()
  
    
    HOURS = [36, 48, 60, 72, 84]

    data_weather_with_months = AddMonthDummies(data_weather)
    data_weather_hours = GetForecastingHours(data_weather_with_months, HOURS)

    data_weather_hours['36'] = data_weather_hours['36'].drop(columns = ["ens_mean_-61", "ens_mean_-59", "ens_mean_-57", "ens_mean_-2", "ens_mean_-1", "ens_mean_1", "ens_mean_2"])
    data_weather_hours['48'] = data_weather_hours['48'].drop(columns = ["ens_mean_-61", "ens_mean_-59", "ens_mean_-49", "ens_mean_-2", "ens_mean_-1", "ens_mean_1", "ens_mean_6"])
    data_weather_hours['60'] = data_weather_hours['60'].drop(columns = ["ens_mean_-61", "ens_mean_-49", "ens_mean_-57", "ens_mean_-6", "ens_mean_-1", "ens_mean_1", "ens_mean_6"])
    data_weather_hours['72'] = data_weather_hours['72'].drop(columns = ["ens_mean_-49", "ens_mean_-59", "ens_mean_-57", "ens_mean_-6", "ens_mean_-1", "ens_mean_6", "ens_mean_2"])
    data_weather_hours['84'] = data_weather_hours['84'].drop(columns = ["ens_mean_-49", "ens_mean_-59", "ens_mean_-57", "ens_mean_-2", "ens_mean_-6", "ens_mean_6", "ens_mean_2"])
    
    data_weather_total = MergeForecastingWindows(data_weather_hours)
    data_weather_total = data_weather_total.sort_values(by=['init_tm', 'fcst_hour']).reset_index(drop = True)

    x_36_cols = data_weather_hours['36'].columns
    x_36_cols = [x for x in x_36_cols if x not in ['init_tm', 'met_var', 'obs', 'ens_mean_-49']]

    x_48_cols = data_weather_hours['48'].columns
    x_48_cols = [x for x in x_48_cols if x not in ['init_tm', 'met_var', 'obs', 'ens_mean_-57']]

    x_60_cols = data_weather_hours['60'].columns
    x_60_cols = [x for x in x_60_cols if x not in ['init_tm', 'met_var', 'obs', 'ens_mean_-59']]

    x_72_cols = data_weather_hours['72'].columns
    x_72_cols = [x for x in x_72_cols if x not in ['init_tm', 'met_var', 'obs', 'ens_mean_-61']]

    x_84_cols = data_weather_hours['84'].columns
    x_84_cols = [x for x in x_84_cols if x not in ['init_tm', 'met_var', 'obs', 'ens_mean_-61']]

    X_36_train, X_36_test, y_36_train, y_36_test = Remove_Outlier_and_Split(data_weather_hours['36'], plot_outliers   = True, columns_X = x_36_cols, columns_y = "obs")
    X_48_train, X_48_test, y_48_train, y_48_test = Remove_Outlier_and_Split(data_weather_hours['48'], plot_outliers   = True, columns_X = x_48_cols, columns_y = "obs")
    X_60_train, X_60_test, y_60_train, y_60_test = Remove_Outlier_and_Split(data_weather_hours['60'], plot_outliers   = True, columns_X = x_60_cols, columns_y = "obs")
    X_72_train, X_72_test, y_72_train, y_72_test = Remove_Outlier_and_Split(data_weather_hours['72'], plot_outliers   = True, columns_X = x_72_cols, columns_y = "obs")
    X_84_train, X_84_test, y_84_train, y_84_test = Remove_Outlier_and_Split(data_weather_hours['84'], plot_outliers   = True, columns_X = x_84_cols, columns_y = "obs")
    X_t_train, X_t_test, y_t_train, y_t_test     = Remove_Outlier_and_Split(data_weather_total,       plot_outliers   = True, columns_X = [data_weather_total.columns[0], data_weather_total.columns[2], *X_36_train.columns[42:44]], columns_y = "obs")
    #X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test
    
    return X_t_train, X_t_test, y_t_train, y_t_test

def Prepare_Other(data_weather):

    print('Shape: ',data_weather.shape)

    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(61)), columns =["ens_mean_-61"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(59)), columns =["ens_mean_-59"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(57)), columns =["ens_mean_-57"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(49)), columns =["ens_mean_-49"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(6)), columns =["ens_mean_-6"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(2)), columns =["ens_mean_-2"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(1)), columns =["ens_mean_-1"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(-1)), columns =["ens_mean_1"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(-2)), columns =["ens_mean_2"]).reset_index(drop = True)], axis=1)
    data_weather = pd.concat([data_weather, pd.DataFrame(list(data_weather["ens_mean"].shift(-6)), columns =["ens_mean_6"]).reset_index(drop = True)], axis=1)

    data_weather = data_weather.dropna()
  
    QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]
    HOURS = [36, 48, 60, 72, 84]

    data_weather_with_months = AddMonthDummies(data_weather)
    data_weather_hours = GetForecastingHours(data_weather_with_months, HOURS)

    data_weather_hours['36'] = data_weather_hours['36'].drop(columns = ["ens_mean_-61", "ens_mean_-59", "ens_mean_-57", "ens_mean_-2", "ens_mean_-1", "ens_mean_1", "ens_mean_2"])
    data_weather_hours['48'] = data_weather_hours['48'].drop(columns = ["ens_mean_-61", "ens_mean_-59", "ens_mean_-49", "ens_mean_-2", "ens_mean_-1", "ens_mean_1", "ens_mean_6"])
    data_weather_hours['60'] = data_weather_hours['60'].drop(columns = ["ens_mean_-61", "ens_mean_-49", "ens_mean_-57", "ens_mean_-6", "ens_mean_-1", "ens_mean_1", "ens_mean_6"])
    data_weather_hours['72'] = data_weather_hours['72'].drop(columns = ["ens_mean_-49", "ens_mean_-59", "ens_mean_-57", "ens_mean_-6", "ens_mean_-1", "ens_mean_6", "ens_mean_2"])
    data_weather_hours['84'] = data_weather_hours['84'].drop(columns = ["ens_mean_-49", "ens_mean_-59", "ens_mean_-57", "ens_mean_-2", "ens_mean_-6", "ens_mean_6", "ens_mean_2"])
    
    x_36_cols = data_weather_hours['36'].columns
    x_36_cols = [x for x in x_36_cols if x not in ['init_tm', 'met_var', 'obs', 'ens_mean_-49']]

    x_48_cols = data_weather_hours['48'].columns
    x_48_cols = [x for x in x_48_cols if x not in ['init_tm', 'met_var', 'obs', 'ens_mean_-57']]

    x_60_cols = data_weather_hours['60'].columns
    x_60_cols = [x for x in x_60_cols if x not in ['init_tm', 'met_var', 'obs', 'ens_mean_-59']]

    x_72_cols = data_weather_hours['72'].columns
    x_72_cols = [x for x in x_72_cols if x not in ['init_tm', 'met_var', 'obs', 'ens_mean_-61']]

    x_84_cols = data_weather_hours['84'].columns
    x_84_cols = [x for x in x_84_cols if x not in ['init_tm', 'met_var', 'obs', 'ens_mean_-61']]

    X_36_train, X_36_test, y_36_train, y_36_test = Remove_Outlier_and_Split(data_weather_hours['36'], plot_outliers   = True, columns_X = x_36_cols, columns_y = "obs")
    X_48_train, X_48_test, y_48_train, y_48_test = Remove_Outlier_and_Split(data_weather_hours['48'], plot_outliers   = True, columns_X = x_48_cols, columns_y = "obs")
    X_60_train, X_60_test, y_60_train, y_60_test = Remove_Outlier_and_Split(data_weather_hours['60'], plot_outliers   = True, columns_X = x_60_cols, columns_y = "obs")
    X_72_train, X_72_test, y_72_train, y_72_test = Remove_Outlier_and_Split(data_weather_hours['72'], plot_outliers   = True, columns_X = x_72_cols, columns_y = "obs")
    X_84_train, X_84_test, y_84_train, y_84_test = Remove_Outlier_and_Split(data_weather_hours['84'], plot_outliers   = True, columns_X = x_84_cols, columns_y = "obs")
   
    return X_36_train, X_36_test, y_36_train, y_36_test, X_48_train, X_48_test, y_48_train, y_48_test, X_60_train, X_60_test, y_60_train, y_60_test,X_72_train, X_72_test, y_72_train, y_72_test,  X_84_train, X_84_test, y_84_train, y_84_test
    
