#%%
# Imports

import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import norm, genhyperbolic
from scipy.stats import rv_continuous
from scipy.stats import studentized_range
import pingouin as pg

from matplotlib import pyplot as plt

from DAX_Preprocessing_Evaluation_Functions import *
from DAX_LSTM_Functions import *
from DAX_Other_Models_Functions import *
from DAX_GARCH_Functions import *

# %%

QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

df_dax_1, df_dax_2, df_dax_3, df_dax_4, df_dax_5 = Prepare(number_datapoints = 1400, skip = 4)
df_dax_1_lstm, df_dax_2_lstm, df_dax_3_lstm, df_dax_4_lstm, df_dax_5_lstm = Prepare(number_datapoints = 500, skip = 4) 

X_train_1, X_test_1, y_train_1, y_test_1 = Remove_Outlier_and_Split(df_dax_1, True, ["x-5", "x-4", "x-3", "x-2", "x-1"], ["y"])
X_train_2, X_test_2, y_train_2, y_test_2 = Remove_Outlier_and_Split(df_dax_2, True, ["x-5", "x-4", "x-3", "x-2", "x-1"], ["y"])
X_train_3, X_test_3, y_train_3, y_test_3 = Remove_Outlier_and_Split(df_dax_3, True, ["x-5", "x-4", "x-3", "x-2", "x-1"], ["y"])
X_train_4, X_test_4, y_train_4, y_test_4 = Remove_Outlier_and_Split(df_dax_4, True, ["x-5", "x-4", "x-3", "x-2", "x-1"], ["y"])
X_train_5, X_test_5, y_train_5, y_test_5 = Remove_Outlier_and_Split(df_dax_5, True, ["x-5", "x-4", "x-3", "x-2", "x-1"], ["y"])

X_train_1_lstm, X_test_1_lstm, y_train_1_lstm, y_test_1_lstm = Remove_Outlier_and_Split_LSTM(df_dax_1_lstm, True, ["x-5", "x-4", "x-3", "x-2", "x-1"], ["y"])
X_train_2_lstm, X_test_2_lstm, y_train_2_lstm, y_test_2_lstm = Remove_Outlier_and_Split_LSTM(df_dax_2_lstm, True, ["x-5", "x-4", "x-3", "x-2", "x-1"], ["y"])
X_train_3_lstm, X_test_3_lstm, y_train_3_lstm, y_test_3_lstm = Remove_Outlier_and_Split_LSTM(df_dax_3_lstm, True, ["x-5", "x-4", "x-3", "x-2", "x-1"], ["y"])
X_train_4_lstm, X_test_4_lstm, y_train_4_lstm, y_test_4_lstm = Remove_Outlier_and_Split_LSTM(df_dax_4_lstm, True, ["x-5", "x-4", "x-3", "x-2", "x-1"], ["y"])
X_train_5_lstm, X_test_5_lstm, y_train_5_lstm, y_test_5_lstm = Remove_Outlier_and_Split_LSTM(df_dax_5_lstm, True, ["x-5", "x-4", "x-3", "x-2", "x-1"], ["y"])


# %%

# Q-Q Plots for year 2021
s = 1
for i in [y_train_1['y'], y_train_2['y'], y_train_3['y'], y_train_4['y'], y_train_5['y']]:

  ax = pg.qqplot(i.iloc[-255:], dist='norm', figsize = (12, 8))
  plt.savefig('QQ Plot 2021 ' + str(s) + '-Step.png', dpi= 300)

  s += 1
# %%

# Distribution Plots

p =     [0,0,0,0,0]
a =     [1.75,0.8,0.6,0.3,0.3]
b =     [0,0,0,0,0]
loc =   [0.05,0.1,0.1,0.1,0.2]
s = 0

for i in [y_train_1['y'], y_train_2['y'], y_train_3['y'], y_train_4['y'], y_train_5['y']]:
    
    fig, ax = plt.subplots(1, 1)
 
    x = np.linspace(genhyperbolic.ppf(0.005, p[s], a[s], b[s],loc[s]),
                    genhyperbolic.ppf(0.995, p[s], a[s], b[s],loc[s]), 
                    1000)

    plt.hist(np.array(i), 
            'auto', 
            color='#333333',
            alpha=0.3, 
            density=True)

    mu = 0
    variance = 1
    sigma = np.sqrt(variance)
    x2 = np.linspace(mu - 5*sigma, 
                     mu + 5*sigma, 
                     1000)

    ax.plot(x, genhyperbolic.pdf(x, p[s], a[s], b[s],loc[s]), 
            label='genhyperbolic pdf', linewidth=1.5)

    ax.plot(x2, stats.norm.pdf(x2, mu, sigma),
            'k-', linewidth=0.5)

    ax.legend([r'Genhyperbolic'+  "\n"+
               r'$\alpha$ = 0.3, $\mu$ = 0.2', 
               'Normal Distribution', 
               'Histogram '+ "\n"+
               r'5-step return'], loc = 'upper left')
    s += 1
    plt.ylabel('Density')
    plt.xlabel('Return in %')
    plt.savefig('Distribution ' + str(s) + '.png', dpi= 300)
    
# %%
