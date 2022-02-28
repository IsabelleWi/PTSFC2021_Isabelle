#%%
# Imports

import pandas as pd
import numpy as np

from Preprocessing_Evaluation_Functions import *
from LSTM_Functions import *
from Models_Functions import *
from GARCH_Functions import *

QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]

#%%

df_dax_1, df_dax_2, df_dax_3, df_dax_4, df_dax_5 = Prepare(number_datapoints = 1406, skip = 12) #Skip needs to be adapted according to day
df_dax_1_lstm, df_dax_2_lstm, df_dax_3_lstm, df_dax_4_lstm, df_dax_5_lstm = Prepare(number_datapoints = 406, skip = 12) #Skip needs to be adapted according to day

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

train_1_x, train_1_y, test_1_x, test_1_y = MakeTorch_for_prediction(pd.concat([X_train_1_lstm, X_test_1_lstm]), pd.concat([y_train_1_lstm, y_test_1_lstm]), pd.DataFrame(X_test_1_lstm.iloc[-1:,:]), pd.DataFrame(y_test_1_lstm.iloc[-1:,:]))
train_2_x, train_2_y, test_2_x, test_2_y = MakeTorch_for_prediction(pd.concat([X_train_2_lstm, X_test_2_lstm]), pd.concat([y_train_2_lstm, y_test_2_lstm]), pd.DataFrame(X_test_2_lstm.iloc[-1:,:]), pd.DataFrame(y_test_2_lstm.iloc[-1:,:]))
train_3_x, train_3_y, test_3_x, test_3_y = MakeTorch_for_prediction(pd.concat([X_train_3_lstm, X_test_3_lstm]), pd.concat([y_train_3_lstm, y_test_3_lstm]), pd.DataFrame(X_test_3_lstm.iloc[-1:,:]), pd.DataFrame(y_test_3_lstm.iloc[-1:,:]))
train_4_x, train_4_y, test_4_x, test_4_y = MakeTorch_for_prediction(pd.concat([X_train_4_lstm, X_test_4_lstm]), pd.concat([y_train_4_lstm, y_test_4_lstm]), pd.DataFrame(X_test_4_lstm.iloc[-1:,:]), pd.DataFrame(y_test_4_lstm.iloc[-1:,:]))
train_5_x, train_5_y, test_5_x, test_5_y = MakeTorch_for_prediction(pd.concat([X_train_5_lstm, X_test_5_lstm]), pd.concat([y_train_5_lstm, y_test_5_lstm]), pd.DataFrame(X_test_5_lstm.iloc[-1:,:]), pd.DataFrame(y_test_5_lstm.iloc[-1:,:]))

#%%

l1 = Train_LSTM(train_1_x, train_1_y, test_1_x, num_epochs = 500, learning_rate = 0.01)
l2 = Train_LSTM(train_2_x, train_2_y, test_2_x, num_epochs = 500, learning_rate = 0.01)
l3 = Train_LSTM(train_3_x, train_3_y, test_3_x, num_epochs = 500, learning_rate = 0.01)
l4 = Train_LSTM(train_4_x, train_4_y, test_4_x, num_epochs = 500, learning_rate = 0.01)
l5 = Train_LSTM(train_5_x, train_5_y, test_5_x, num_epochs = 500, learning_rate = 0.01)

#%%
g1_l = TrainGarchNormal(pd.concat([y_train_1, y_test_1]), mean =[l1.iloc[0,0]], alpha = 0.35, beta = 0.05, test_size = 1)
g3_l = TrainGarchNormal(pd.concat([y_train_3, y_test_3]), mean =[l3.iloc[0,0]], alpha = 0.35, beta = 0.05, test_size = 1)
g2_l = TrainGarchNormal(pd.concat([y_train_2, y_test_2]), mean =[l2.iloc[0,0]], alpha = 0.35, beta = 0.05, test_size = 1)
g4_l = TrainGarchNormal(pd.concat([y_train_4, y_test_4]), mean =[l4.iloc[0,0]], alpha = 0.35, beta = 0.05, test_size = 1)
g5_l = TrainGarchNormal(pd.concat([y_train_5, y_test_5]), mean =[l5.iloc[0,0]], alpha = 0.35, beta = 0.05, test_size = 1)

#%%
df_gl = pd.concat([g1_l, g2_l, g3_l, g4_l, g5_l], axis = 0)
#%%
df_gl.to_excel("Dax Prediciton.xlsx")

# %%
