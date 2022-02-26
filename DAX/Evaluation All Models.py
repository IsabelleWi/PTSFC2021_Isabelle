#%%
# Imports

import pandas as pd
import numpy as np

from DAX_Preprocessing_Evaluation_Functions import *
from DAX_LSTM_Functions import *
from DAX_Other_Models_Functions import *
from DAX_GARCH_Functions import *

QUANTILES = [0.025, 0.25, 0.5, 0.75, 0.975]
#%%
df_dax_1, df_dax_2, df_dax_3, df_dax_4, df_dax_5 = Prepare(number_datapoints = 1400, skip = 5) #Skip needs to be adapted according to business days that past, Results were created on 25th Februar with skip 4
df_dax_1_lstm, df_dax_2_lstm, df_dax_3_lstm, df_dax_4_lstm, df_dax_5_lstm = Prepare(number_datapoints = 500, skip = 5) #Skip needs to be adapted according to business days that past, Results were created on 25th Februar with skip 4

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

train_1_x, train_1_y, test_1_x, test_1_y = MakeTorch(X_train_1_lstm, y_train_1_lstm, X_test_1_lstm, y_test_1_lstm)
train_2_x, train_2_y, test_2_x, test_2_y = MakeTorch(X_train_2_lstm, y_train_2_lstm, X_test_2_lstm, y_test_2_lstm)
train_3_x, train_3_y, test_3_x, test_3_y = MakeTorch(X_train_3_lstm, y_train_3_lstm, X_test_3_lstm, y_test_3_lstm)
train_4_x, train_4_y, test_4_x, test_4_y = MakeTorch(X_train_4_lstm, y_train_4_lstm, X_test_4_lstm, y_test_4_lstm)
train_5_x, train_5_y, test_5_x, test_5_y = MakeTorch(X_train_5_lstm, y_train_5_lstm, X_test_5_lstm, y_test_5_lstm)
#%%

o1 = OLS_with_Quantiles(X_train_1, y_train_1, X_test_1, y_test_1)
o2 = OLS_with_Quantiles(X_train_2, y_train_2, X_test_2, y_test_2)
o3 = OLS_with_Quantiles(X_train_3, y_train_3, X_test_3, y_test_3)
o4 = OLS_with_Quantiles(X_train_4, y_train_4, X_test_4, y_test_4)
o5 = OLS_with_Quantiles(X_train_5, y_train_5, X_test_5, y_test_5)

qr1 = QuantReg(X_train_1, y_train_1, X_test_1, y_test_1)
qr2 = QuantReg(X_train_2, y_train_2, X_test_2, y_test_2)
qr3 = QuantReg(X_train_3, y_train_3, X_test_3, y_test_3)
qr4 = QuantReg(X_train_4, y_train_4, X_test_4, y_test_4)
qr5 = QuantReg(X_train_5, y_train_5, X_test_5, y_test_5)

rf1 = RF_with_Quantiles(X_train_1, y_train_1, X_test_1, y_test_1)
rf2 = RF_with_Quantiles(X_train_2, y_train_2, X_test_2, y_test_2)
rf3 = RF_with_Quantiles(X_train_3, y_train_3, X_test_3, y_test_3)
rf4 = RF_with_Quantiles(X_train_4, y_train_4, X_test_4, y_test_4)
rf5 = RF_with_Quantiles(X_train_5, y_train_5, X_test_5, y_test_5)

gb1 = GB_Quantiles(X_train_1, y_train_1, X_test_1, y_test_1)
gb2 = GB_Quantiles(X_train_2, y_train_2, X_test_2, y_test_2)
gb3 = GB_Quantiles(X_train_3, y_train_3, X_test_3, y_test_3)
gb4 = GB_Quantiles(X_train_4, y_train_4, X_test_4, y_test_4)
gb5 = GB_Quantiles(X_train_5, y_train_5, X_test_5, y_test_5)
#%%
l1 = Train_LSTM(train_1_x, train_1_y, test_1_x, num_epochs = 500, learning_rate = 0.01)
l2 = Train_LSTM(train_2_x, train_2_y, test_2_x, num_epochs = 500, learning_rate = 0.01)
l3 = Train_LSTM(train_3_x, train_3_y, test_3_x, num_epochs = 500, learning_rate = 0.01)
l4 = Train_LSTM(train_4_x, train_4_y, test_4_x, num_epochs = 500, learning_rate = 0.01)
l5 = Train_LSTM(train_5_x, train_5_y, test_5_x, num_epochs = 500, learning_rate = 0.01)

g1 = TrainGarchNormal(pd.concat([y_train_1, y_test_1]), mean =[0]*50, test_size = len(y_test_1))
g2 = TrainGarchNormal(pd.concat([y_train_2, y_test_2]), mean =[0]*50, test_size = len(y_test_2))
g3 = TrainGarchNormal(pd.concat([y_train_3, y_test_3]), mean =[0]*50, test_size = len(y_test_3))
g4 = TrainGarchNormal(pd.concat([y_train_4, y_test_4]), mean =[0]*50, test_size = len(y_test_4))
g5 = TrainGarchNormal(pd.concat([y_train_5, y_test_5]), mean =[0]*50, test_size = len(y_test_5))

g1_l = TrainGarchNormal(pd.concat([y_train_1, y_test_1]), mean =l1.iloc[:,2:].iloc[0,:].reset_index(drop=True), alpha = 0.2, test_size = len(y_test_1))
g2_l = TrainGarchNormal(pd.concat([y_train_2, y_test_2]), mean =l2.iloc[:,1:].iloc[0,:].reset_index(drop=True), alpha = 0.2, test_size = len(y_test_2))
g3_l = TrainGarchNormal(pd.concat([y_train_3, y_test_3]), mean =l3.iloc[:,1:].iloc[0,:].reset_index(drop=True), alpha = 0.2, test_size = len(y_test_3))
g4_l = TrainGarchNormal(pd.concat([y_train_4, y_test_4]), mean =l4.iloc[:,1:].iloc[0,:].reset_index(drop=True), alpha = 0.2, test_size = len(y_test_4))
g5_l = TrainGarchNormal(pd.concat([y_train_5, y_test_5]), mean =l5.iloc[:, :].iloc[0,:].reset_index(drop=True), alpha = 0.2, test_size = len(y_test_5))

g1_gen = TrainGarchGenhyp(pd.concat([y_train_1, y_test_1]), mean =[0]*50, alpha = 0.35, test_size = len(y_test_1))
g2_gen = TrainGarchGenhyp(pd.concat([y_train_2, y_test_2]), mean =[0]*50, alpha = 0.35, test_size = len(y_test_2))
g3_gen = TrainGarchGenhyp(pd.concat([y_train_3, y_test_3]), mean =[0]*50, alpha = 0.35, test_size = len(y_test_3))
g4_gen = TrainGarchGenhyp(pd.concat([y_train_4, y_test_4]), mean =[0]*50, alpha = 0.35, test_size = len(y_test_4))
g5_gen = TrainGarchGenhyp(pd.concat([y_train_5, y_test_5]), mean =[0]*50, alpha = 0.35, test_size = len(y_test_5))

g1_l_gen = TrainGarchGenhyp(pd.concat([y_train_1, y_test_1]), mean =l1.iloc[:,2:].iloc[0,:].reset_index(drop=True), alpha = 0.2, test_size = len(y_test_1))
g2_l_gen = TrainGarchGenhyp(pd.concat([y_train_2, y_test_2]), mean =l2.iloc[:,1:].iloc[0,:].reset_index(drop=True), alpha = 0.2, test_size = len(y_test_2))
g3_l_gen = TrainGarchGenhyp(pd.concat([y_train_3, y_test_3]), mean =l3.iloc[:,1:].iloc[0,:].reset_index(drop=True), alpha = 0.2, test_size = len(y_test_3))
g4_l_gen = TrainGarchGenhyp(pd.concat([y_train_4, y_test_4]), mean =l4.iloc[:,1:].iloc[0,:].reset_index(drop=True), alpha = 0.2, test_size = len(y_test_4))
g5_l_gen = TrainGarchGenhyp(pd.concat([y_train_5, y_test_5]), mean =l5.iloc[:, :].iloc[0,:].reset_index(drop=True), alpha = 0.2, test_size = len(y_test_5))
#%%
R_O_1 = Backtesting_per_Timestep(o1, y_test_1.reset_index(drop=True), '1')
R_O_2 = Backtesting_per_Timestep(o2, y_test_2.reset_index(drop=True), '2')
R_O_3 = Backtesting_per_Timestep(o3, y_test_3.reset_index(drop=True), '3')
R_O_4 = Backtesting_per_Timestep(o4, y_test_4.reset_index(drop=True), '4')
R_O_5 = Backtesting_per_Timestep(o5, y_test_5.reset_index(drop=True), '5')

R_QR_1 = Backtesting_per_Timestep(qr1, y_test_1.reset_index(drop=True), '1')
R_QR_2 = Backtesting_per_Timestep(qr2, y_test_2.reset_index(drop=True), '2')
R_QR_3 = Backtesting_per_Timestep(qr3, y_test_3.reset_index(drop=True), '3')
R_QR_4 = Backtesting_per_Timestep(qr4, y_test_4.reset_index(drop=True), '4')
R_QR_5 = Backtesting_per_Timestep(qr5, y_test_5.reset_index(drop=True), '5')

R_RF_1 =Backtesting_per_Timestep(rf1, y_test_1.reset_index(drop=True), '1')
R_RF_2 =Backtesting_per_Timestep(rf2, y_test_2.reset_index(drop=True), '2')
R_RF_3 =Backtesting_per_Timestep(rf3, y_test_3.reset_index(drop=True), '3')
R_RF_4 =Backtesting_per_Timestep(rf4, y_test_4.reset_index(drop=True), '4')
R_RF_5 =Backtesting_per_Timestep(rf5, y_test_5.reset_index(drop=True), '5')

R_QB_1 = Backtesting_per_Timestep(gb1, y_test_1.reset_index(drop=True), '1')
R_QB_2 = Backtesting_per_Timestep(gb2, y_test_2.reset_index(drop=True), '2')
R_QB_3 = Backtesting_per_Timestep(gb3, y_test_3.reset_index(drop=True), '3')
R_QB_4 = Backtesting_per_Timestep(gb4, y_test_4.reset_index(drop=True), '4')
R_QB_5 = Backtesting_per_Timestep(gb5, y_test_5.reset_index(drop=True), '5')

ga1 = Backtesting_per_Timestep(g1, y_test_1.reset_index(drop=True), '1')
ga2 = Backtesting_per_Timestep(g2, y_test_2.reset_index(drop=True), '2')
ga3 = Backtesting_per_Timestep(g3, y_test_3.reset_index(drop=True), '3')
ga4 = Backtesting_per_Timestep(g4, y_test_4.reset_index(drop=True), '4')
ga5 = Backtesting_per_Timestep(g5, y_test_5.reset_index(drop=True), '5')

gl1 = Backtesting_per_Timestep(g1_l, y_test_1.reset_index(drop=True), '1')
gl2 = Backtesting_per_Timestep(g2_l, y_test_2.reset_index(drop=True), '2')
gl3 = Backtesting_per_Timestep(g3_l, y_test_3.reset_index(drop=True), '3')
gl4 = Backtesting_per_Timestep(g4_l, y_test_4.reset_index(drop=True), '4')
gl5 = Backtesting_per_Timestep(g5_l, y_test_5.reset_index(drop=True), '5')

ga1_gen = Backtesting_per_Timestep(g1_gen, y_test_1.reset_index(drop=True), '1')
ga2_gen = Backtesting_per_Timestep(g2_gen, y_test_2.reset_index(drop=True), '2')
ga3_gen = Backtesting_per_Timestep(g3_gen, y_test_3.reset_index(drop=True), '3')
ga4_gen = Backtesting_per_Timestep(g4_gen, y_test_4.reset_index(drop=True), '4')
ga5_gen = Backtesting_per_Timestep(g5_gen, y_test_5.reset_index(drop=True), '5')

gl1_gen = Backtesting_per_Timestep(g1_l_gen, y_test_1.reset_index(drop=True), '1')
gl2_gen = Backtesting_per_Timestep(g2_l_gen, y_test_2.reset_index(drop=True), '2')
gl3_gen = Backtesting_per_Timestep(g3_l_gen, y_test_3.reset_index(drop=True), '3')
gl4_gen = Backtesting_per_Timestep(g4_l_gen, y_test_4.reset_index(drop=True), '4')
gl5_gen = Backtesting_per_Timestep(g5_l_gen, y_test_5.reset_index(drop=True), '5')

df_o = pd.concat([R_O_1, R_O_2, R_O_3, R_O_4, R_O_5], axis = 1)
df_rf = pd.concat([R_RF_1, R_RF_2, R_RF_3, R_RF_4, R_RF_5], axis = 1)
df_qr = pd.concat([R_QR_1, R_QR_2, R_QR_3, R_QR_4, R_QR_5], axis = 1)
df_qb = pd.concat([R_QB_1, R_QB_2, R_QB_3, R_QB_4, R_QB_5], axis = 1)
df_ga = pd.concat([ga1, ga2, ga3, ga4, ga5], axis = 1)
df_gl = pd.concat([gl1, gl2, gl3, gl4, gl5], axis = 1)
df_ga_gen = pd.concat([ga1_gen, ga2_gen, ga3_gen, ga4_gen, ga5_gen], axis = 1)
df_gl_gen = pd.concat([gl1_gen, gl2_gen, gl3_gen, gl4_gen, gl5_gen], axis = 1)

df_o['Average'] = df_o.mean(axis=1)
df_rf['Average'] = df_rf.mean(axis=1)
df_qr['Average'] = df_qr.mean(axis=1)
df_qb['Average'] = df_qb.mean(axis=1)
df_ga['Average'] = df_ga.mean(axis=1)
df_gl['Average'] = df_gl.mean(axis=1)
df_ga_gen['Average'] = df_ga_gen.mean(axis=1)
df_gl_gen['Average'] = df_gl_gen.mean(axis=1)

RESULTS = pd.concat([df_o, df_rf, df_qr, df_qb, df_ga, df_gl, df_ga_gen, df_gl_gen], 
                    keys = ['OLS', 'Random Forest', 'Quantile  Regression', 'Gradient Boosting', 'Garch (Normal)', 'Garch (Normal) with LSTM', 'Garch (GenHyp)', 'Garch (GenHyp) with LSTM'], axis = 1)

RESULTS.to_excel('RESULTS.xlsx')
# %%
