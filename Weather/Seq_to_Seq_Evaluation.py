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

# %%
df_temp = pd.read_excel("Temp_Input.xlsx")
df_wind = pd.read_excel("Wind_Input.xlsx")

df_temp = df_temp.iloc[:,1:]
df_wind = df_wind.iloc[:,1:]

X_train_temp, X_test_temp, y_train_temp, y_test_temp = Prepare_NN(df_temp)
X_train_wind, X_test_wind, y_train_wind, y_test_wind = Prepare_NN(df_wind)

# %%

teacher_forcing_ratio_temp    = 0.1 
learning_rate_temp            = 0.01
teacher_forcing_ratio_wind_1  = 0.2 
teacher_forcing_ratio_wind_2  = 0.15 
learning_rate_wind_1          = 0.01
learning_rate_wind_2          = 0.005

l1_t, l2_t, l3_t, l4_t, l5_t = Train_all_NNs(X_train_temp, X_test_temp, y_train_temp, y_test_temp, 'Temp', teacher_forcing_ratio_temp, teacher_forcing_ratio_temp, learning_rate_temp, learning_rate_temp)
l1_w, l2_w, l3_w, l4_w, l5_w = Train_all_NNs(X_train_wind, X_test_wind, y_train_wind, y_test_wind, 'Wind', teacher_forcing_ratio_wind_1, teacher_forcing_ratio_wind_2, learning_rate_wind_1, learning_rate_wind_2)

# %%
plt.figure(figsize=(7,5))
plt.plot(pd.DataFrame(l1_t).iloc[0:], label = 'LSTM')
plt.plot(pd.DataFrame(l2_t).iloc[0:], label = 'Seq-to-Features')
plt.plot(pd.DataFrame(l3_t).iloc[0:], label = 'Seq-to-Seq')
plt.plot(pd.DataFrame(l4_t).iloc[0:], label = 'Seq-to-Features reversed')
plt.plot(pd.DataFrame(l5_t).iloc[0:], label = 'Seq-to-Seq reversed')
plt.legend(loc='upper right')
plt.ylabel("MSE Loss per Epoch")
plt.ylim((0,12))
plt.xlabel("Epoch")
plt.savefig('NN_Temp_loss.png', dpi= 300)
plt.show()

# %%
plt.figure(figsize=(7,5))
plt.plot(pd.DataFrame(l1_w).iloc[0:], label = 'LSTM')
plt.plot(pd.DataFrame(l2_w).iloc[0:], label = 'Seq-to-Features')
plt.plot(pd.DataFrame(l3_w).iloc[0:], label = 'Seq-to-Seq')
plt.plot(pd.DataFrame(l4_w).iloc[0:], label = 'Seq-to-Features reversed')
plt.plot(pd.DataFrame(l5_w).iloc[0:], label = 'Seq-to-Seq reversed')
plt.legend(loc='upper right')
plt.ylabel("MSE Loss per Epoch")
plt.ylim((0,52))
plt.xlabel("Epoch")
plt.savefig('NN_Wind_loss.png', dpi= 300)
plt.show()

# %%
