# Imports

import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.metrics import mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt


def Average(lst):

    """
    Calculate Average of a List/ Pandas Dataframe
    """

    return sum(lst) / len(lst)

def compute_return(adj_close, h):
  
  """
  Computes log return

  Input:
  adj_close (Pandas DataFrame): Timeseries of Adjusted Close 
  h(int) : Stepsize
  """

  n = len(adj_close)
  y2 = adj_close.drop(range(0,h)).reset_index(drop=True) # exclude first h observations
  y1 = adj_close.drop(range(n-h,n)).reset_index(drop=True) # exclude last h observations

  Calc = 100*(np.log(y2)- np.log(y1))

  log_ret = pd.DataFrame([Calc])
  na = pd.DataFrame([["NA"]*h])

  ret = pd.concat([na, log_ret], axis = 1)

  return ret.iloc[1,:]



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


def Remove_Outlier_and_Split_LSTM(df_input, 
                                plot_outliers,
                                columns_X,
                                columns_y):
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
  df_3sigma["std_delivery_time"]=pd.DataFrame((len(df_input[columns_y])+1) * [df_3sigma[columns_y].apply("std")])
  df_3sigma["mean_delivery_time"]=pd.DataFrame((len(df_input[columns_y])+1) * [df_3sigma[columns_y].apply("mean")])

  
  df_3sigma["three_sigma_obs"]=1*df_3sigma["std_delivery_time"]
  df_3sigma["threshold_obs_upper_limit"]= df_3sigma["mean_delivery_time"] + df_3sigma["three_sigma_obs"]
  df_3sigma["threshold_obs_lower_limit"]= df_3sigma["mean_delivery_time"] - df_3sigma["three_sigma_obs"]
  
  df_3sigma["Is_outlier"]=np.where((df_3sigma.y> df_3sigma.threshold_obs_upper_limit)| (df_3sigma.y < df_3sigma.threshold_obs_lower_limit) , True, False)

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


def Remove_Outlier_and_Split(df_input, 
                            plot_outliers,
                            columns_X,
                            columns_y):
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
  df_3sigma["std_delivery_time"]=pd.DataFrame((len(df_input[columns_y])+1) * [df_3sigma[columns_y].apply("std")])
  df_3sigma["mean_delivery_time"]=pd.DataFrame((len(df_input[columns_y])+1) * [df_3sigma[columns_y].apply("mean")])

  

  df_3sigma["three_sigma_obs"]=3*df_3sigma["std_delivery_time"]
  df_3sigma["threshold_obs_upper_limit"]= df_3sigma["mean_delivery_time"] + df_3sigma["three_sigma_obs"]
  df_3sigma["threshold_obs_lower_limit"]= df_3sigma["mean_delivery_time"] - df_3sigma["three_sigma_obs"]
  
  df_3sigma["Is_outlier"]=np.where((df_3sigma.y> df_3sigma.threshold_obs_upper_limit)| (df_3sigma.y < df_3sigma.threshold_obs_lower_limit) , True, False)

  df_train = df_3sigma[:int(df_3sigma.shape[0]*0.9645)] #drop outliers
  df_train_noOutliers = df_train[df_train.Is_outlier == False]
  df_test = df_3sigma[int(df_3sigma.shape[0]*0.9645):] #leave outliers

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


def Prepare(number_datapoints, skip):

    """
    Input:
    number_datapoints: Number of datapoints to include
    skip: Number of recent datapoints to leave out - Needs to be adapted for consistent train & test data 


    Output:
    Five Dataframes containing 1-step,..,5-step return data
    """

    #Read in Adjusted Closing Data Dax

    dax = yf.Ticker("^GDAXI")
    dax_max = dax.history(period='max', interval='1d', auto_adjust=False)
    dax_max_adjc = pd.DataFrame(dax_max['Adj Close'])
    dax_max_adjc = dax_max_adjc.iloc[:-skip,:]

    print('There are {} number of days in the dataset.'.format(dax_max_adjc.shape[0]))

    #Decide how many observations you want to include
    dax_max_adjc_s = dax_max_adjc[len(dax_max_adjc) - number_datapoints:].reset_index().iloc[:,1]
    
    df_dax_1 = pd.DataFrame([compute_return(dax_max_adjc_s, h=1).reset_index(drop = True).shift(5), compute_return(dax_max_adjc_s, h=1).reset_index(drop = True).shift(4), compute_return(dax_max_adjc_s, h=1).reset_index(drop = True).shift(3), compute_return(dax_max_adjc_s, h=1).reset_index(drop = True).shift(2), compute_return(dax_max_adjc_s, h=1).reset_index(drop = True).shift(1), compute_return(dax_max_adjc_s, h=1).reset_index(drop = True)], index = ["x-5", "x-4", "x-3", "x-2", "x-1", "y"]).T
    df_dax_2 = pd.DataFrame([compute_return(dax_max_adjc_s, h=2).reset_index(drop = True).shift(10), compute_return(dax_max_adjc_s, h=2).reset_index(drop = True).shift(8), compute_return(dax_max_adjc_s, h=2).reset_index(drop = True).shift(6), compute_return(dax_max_adjc_s, h=2).reset_index(drop = True).shift(4), compute_return(dax_max_adjc_s, h=2).reset_index(drop = True).shift(2), compute_return(dax_max_adjc_s, h=2).reset_index(drop = True)], index = ["x-5", "x-4", "x-3", "x-2", "x-1", "y"]).T
    df_dax_3 = pd.DataFrame([compute_return(dax_max_adjc_s, h=3).reset_index(drop = True).shift(15), compute_return(dax_max_adjc_s, h=3).reset_index(drop = True).shift(12), compute_return(dax_max_adjc_s, h=3).reset_index(drop = True).shift(9), compute_return(dax_max_adjc_s, h=3).reset_index(drop = True).shift(6), compute_return(dax_max_adjc_s, h=3).reset_index(drop = True).shift(3), compute_return(dax_max_adjc_s, h=3).reset_index(drop = True)], index = ["x-5", "x-4", "x-3", "x-2", "x-1", "y"]).T
    df_dax_4 = pd.DataFrame([compute_return(dax_max_adjc_s, h=4).reset_index(drop = True).shift(20), compute_return(dax_max_adjc_s, h=4).reset_index(drop = True).shift(16), compute_return(dax_max_adjc_s, h=4).reset_index(drop = True).shift(12), compute_return(dax_max_adjc_s, h=4).reset_index(drop = True).shift(8), compute_return(dax_max_adjc_s, h=4).reset_index(drop = True).shift(4), compute_return(dax_max_adjc_s, h=4).reset_index(drop = True)], index = ["x-5", "x-4", "x-3", "x-2", "x-1", "y"]).T
    df_dax_5 = pd.DataFrame([compute_return(dax_max_adjc_s, h=5).reset_index(drop = True).shift(25), compute_return(dax_max_adjc_s, h=5).reset_index(drop = True).shift(20), compute_return(dax_max_adjc_s, h=5).reset_index(drop = True).shift(15), compute_return(dax_max_adjc_s, h=5).reset_index(drop = True).shift(10), compute_return(dax_max_adjc_s, h=5).reset_index(drop = True).shift(5), compute_return(dax_max_adjc_s, h=5).reset_index(drop = True)], index = ["x-5", "x-4", "x-3", "x-2", "x-1", "y"]).T

    df_dax_1 = df_dax_1.dropna()
    df_dax_2 = df_dax_2.dropna()
    df_dax_3 = df_dax_3.dropna()
    df_dax_4 = df_dax_4.dropna()
    df_dax_5 = df_dax_5.dropna()

    return df_dax_1, df_dax_2, df_dax_3, df_dax_4, df_dax_5 



def Backtesting_per_Timestep(y_pred, y_true, timestep = 'Not given', print_statistics = True, plot_data = True):

  """ 
  Input: (for one forecasted timestep)
  y_pred: Pandas Dataframe n x 5 with n being the number of predicted datapoints with 5 Quantiles (2.5%, 25%, 50%, 75%, 97.5%)
  y_true: Pandas Dataframe containing observed Data n x 1
  timestep: String indentifying forecasted timestep

  Output:
  Score: Sum of Quantile Scores over all n datapoints
  """

  shape = y_pred.shape

  print('-------------------------------------------------------------')
  print('Timestep: ' + str(timestep))
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

    Score[str(QUANTILES[i]) + " QScore Sum"] =  sum([calculate_QuantileScore(QUANTILES[i], y_pred.reset_index(drop=True).iloc[j,i], y_true.reset_index(drop=True).iloc[j,0]) for j in range(0,shape[0])])
    Score[str(QUANTILES[i]) + " QScore Avg"] = Score[str(QUANTILES[i]) + " QScore Sum"]/shape[0]

  Score['QScore Sum'] = sum(Score[str(QUANTILES[i]) + " QScore Avg"] for i in range(0,5))
  Score['QScore Avg'] = sum(Score[str(QUANTILES[i]) + " QScore Avg"] for i in range(0,5))/5
  Score['MAE'] = mean_absolute_error(y_true.reset_index(drop=True).iloc[:,0], y_pred.iloc[:,2])
  Score['Avg Width 95% Quantile'] = Average(y_pred.iloc[:,4] - y_pred.iloc[:,0])
  Score['Avg Width 50% Quantile'] = Average(y_pred.iloc[:,3] - y_pred.iloc[:,1])
  Score['Avg Coverage 95% Quantile'] = Coverage(y_true.reset_index(drop=True).iloc[:,0], y_pred.iloc[:,4], y_pred.iloc[:,0])
  Score['Avg Coverage 50% Quantile'] = Coverage(y_true.reset_index(drop=True).iloc[:,0], y_pred.iloc[:,3], y_pred.iloc[:,1])

  Scores = pd.DataFrame([Score['QScore Sum'], Score['QScore Avg'], Score[str(QUANTILES[0]) + " QScore Avg"], Score[str(QUANTILES[1]) + " QScore Avg"], Score[str(QUANTILES[2]) + " QScore Avg"], Score[str(QUANTILES[3]) + " QScore Avg"], Score[str(QUANTILES[4]) + " QScore Avg"], Score['MAE'], Score['Avg Width 95% Quantile'], Score['Avg Width 50% Quantile'], Score['Avg Coverage 95% Quantile'], Score['Avg Coverage 50% Quantile']], 
                        index = ['QScore Sum', 'QScore Avg', 'QScore 0.025 Avg', 'QScore 0.25 Avg', 'QScore 0.5 Avg', 'QScore 0.75 Avg', 'QScore 0.975 Avg', 'MAE', 'Avg Width 95% Quantile', 'Avg Width 50% Quantile', 'Avg Coverage 95% Quantile', 'Avg Coverage 50% Quantile'], columns = [timestep])

  if print_statistics == True:
    print()
    print(Scores)
    print()
    print('-------------------------------------------------------------')

  return Scores


def IncludeNewAsset (Ticker, NumberTestDays, LengthDAX):

  """
  Input:
  Ticker (String): Yahoo Ticker
  NumberTestDays (Int): Number of Test Days 
  LengthDAX (Int): Length of DAX Data Set (Train and Test)

  Output:
  d (Dict): Dict containing Train, Test, Train_x, Train_y, Test_x, Test_y Data
  """
  
  ticker = yf.Ticker(Ticker)
  data_max = ticker.history(period='max', interval='1d', auto_adjust=False)
  data_max_adjc = pd.DataFrame(data_max['Adj Close'])
  
  #Split in Train and Test Set
  
  data_max_adjc_test = data_max_adjc[len(data_max_adjc) - NumberTestDays:].reset_index().iloc[:,1]
  data_max_adjc_train = data_max_adjc[(len(data_max_adjc)- LengthDAX):len(data_max_adjc) - NumberTestDays].reset_index().iloc[:,1]

  #Calculate Returns

  d = {}

  for i in range(1,6):
    d["ret"+str(i)+"_train"] = compute_return(data_max_adjc_train, h=i)
    d["ret"+str(i)+"_test"] = compute_return(data_max_adjc_test, h=i)

    d["ret" + str(i) + "_train_x"] = d["ret" + str(i) +"_train"].reset_index()[i:].iloc[:-i,1]
    d["ret" + str(i) + "_train_y"] = d["ret" + str(i) +"_train"].reset_index()[i:].iloc[i:,1]

    d["ret" + str(i) + "_test_x"] = d["ret" + str(i) +"_test"].reset_index()[i:].iloc[:-i,1]
    d["ret" + str(i) + "_test_y"] = d["ret" + str(i) +"_test"].reset_index()[i:].iloc[i:,1]

  print('There are {} number of days in the original dataset.'.format(data_max_adjc.shape[0]))
  print('There are {} number of days in the train dataset.'.format(d["ret" + str(i) + "_train"].shape[0])) 
  print('There are {} number of days in the test dataset.'.format(d["ret" + str(i) + "_test"].shape[0]))

  return d


def get_technical_indicators(dataset):

  """
  Input: Pandas DataFrame 1d
  """

  df = pd.DataFrame(dataset.values, columns = ['return'])

  df['MA_15'] = df['return'].transform(lambda x: x.rolling(window = 15).mean())
  df['15Ewm'] = df['return'].transform(lambda x: x.ewm(span=15, adjust=False).mean())
  df['SD'] = df['return'].transform(lambda x: x.rolling(window=15).std())
  df['RC'] = df['return'].transform(lambda x: x.pct_change(periods = 15)) 

  df['RC'].replace([np.inf, -np.inf, np.nan], 0)
  df['15Ewm'].replace([np.inf, -np.inf, np.nan], 0)
  df['SD'].replace([np.inf, -np.inf, np.nan], 0)

  return df


def Remove_Outlier_and_Split_LSTM_for_prediciton(df_input, 
                                plot_outliers,
                                columns_X,
                                columns_y):
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
  df_3sigma["std_delivery_time"]=pd.DataFrame((len(df_input[columns_y])+1) * [df_3sigma[columns_y].apply("std")])
  df_3sigma["mean_delivery_time"]=pd.DataFrame((len(df_input[columns_y])+1) * [df_3sigma[columns_y].apply("mean")])

  
  df_3sigma["three_sigma_obs"]=1*df_3sigma["std_delivery_time"]
  df_3sigma["threshold_obs_upper_limit"]= df_3sigma["mean_delivery_time"] + df_3sigma["three_sigma_obs"]
  df_3sigma["threshold_obs_lower_limit"]= df_3sigma["mean_delivery_time"] - df_3sigma["three_sigma_obs"]
  
  df_3sigma["Is_outlier"]=np.where((df_3sigma.y> df_3sigma.threshold_obs_upper_limit)| (df_3sigma.y < df_3sigma.threshold_obs_lower_limit) , True, False)

  df_train = df_3sigma[:int(df_3sigma.shape[0]*0.99)] #drop outliers
  df_train_noOutliers = df_train[df_train.Is_outlier == False]
  df_test = df_3sigma[int(df_3sigma.shape[0]*0.99):] #leave outliers

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


def Remove_Outlier_and_Split_for_prediciton(df_input, 
                            plot_outliers,
                            columns_X,
                            columns_y):
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
  df_3sigma["std_delivery_time"]=pd.DataFrame((len(df_input[columns_y])+1) * [df_3sigma[columns_y].apply("std")])
  df_3sigma["mean_delivery_time"]=pd.DataFrame((len(df_input[columns_y])+1) * [df_3sigma[columns_y].apply("mean")])

  

  df_3sigma["three_sigma_obs"]=3*df_3sigma["std_delivery_time"]
  df_3sigma["threshold_obs_upper_limit"]= df_3sigma["mean_delivery_time"] + df_3sigma["three_sigma_obs"]
  df_3sigma["threshold_obs_lower_limit"]= df_3sigma["mean_delivery_time"] - df_3sigma["three_sigma_obs"]
  
  df_3sigma["Is_outlier"]=np.where((df_3sigma.y> df_3sigma.threshold_obs_upper_limit)| (df_3sigma.y < df_3sigma.threshold_obs_lower_limit) , True, False)

  df_train = df_3sigma[:int(df_3sigma.shape[0]*0.99)] #drop outliers
  df_train_noOutliers = df_train[df_train.Is_outlier == False]
  df_test = df_3sigma[int(df_3sigma.shape[0]*0.99):] #leave outliers

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
