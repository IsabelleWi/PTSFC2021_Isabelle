import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm
tqdm.pandas()

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from tqdm import trange

from Preprocessing_Evaluation_Functions import *


def PrepareforNets(X_train, X_test, y_train, y_test):

  # Merge X and y datasets to be able to sort data while ensuring matching of y labels stays right
  t_train   = pd.concat([X_train, y_train], axis = 1)
  t_test    = pd.concat([X_test, y_test], axis = 1)

  # Get Init Dates to group data daywise
  column_values_train = X_train[["init_tm"]].values.ravel()
  unique_values_train = pd.unique(column_values_train)

  column_values_test = X_test[["init_tm"]].values.ravel()
  unique_values_test = pd.unique(column_values_test)

  # Group by Init Dates
  grouped_tr = t_train.groupby(t_train['init_tm'])
  grouped_train = {}

  grouped_te = t_test.groupby(t_test['init_tm'])
  grouped_test = {}

  for i in unique_values_train: 
    grouped_train[str(i)] = grouped_tr.get_group(i)

  for i in unique_values_test: 
    grouped_test[str(i)] = grouped_te.get_group(i)

  X_train_asc, y_train_asc, X_train_desc, y_train_desc, X_test_asc, y_test_asc, X_test_desc, y_test_desc  = [], [], [], [], [], [], [], []

  X_test_asc_var, X_test_desc_var = [], []


  for i in range(0, len(grouped_train)):

    if len(grouped_train[str(unique_values_train[i])]) == 5: # Only include Init Dates for which all five horizonts are available
    
      x_asc = np.array([grouped_train[str(unique_values_train[i])]['ens_mean'].iloc[0], 
                        grouped_train[str(unique_values_train[i])]['ens_mean'].iloc[1], 
                        grouped_train[str(unique_values_train[i])]['ens_mean'].iloc[2], 
                        grouped_train[str(unique_values_train[i])]['ens_mean'].iloc[3], 
                        grouped_train[str(unique_values_train[i])]['ens_mean'].iloc[4]])
      
      X_train_asc.append(x_asc)

      x_desc = np.array([grouped_train[str(unique_values_train[i])]['ens_mean'].iloc[4], 
                         grouped_train[str(unique_values_train[i])]['ens_mean'].iloc[3], 
                         grouped_train[str(unique_values_train[i])]['ens_mean'].iloc[2], 
                         grouped_train[str(unique_values_train[i])]['ens_mean'].iloc[1], 
                         grouped_train[str(unique_values_train[i])]['ens_mean'].iloc[0]])
      
      X_train_desc.append(x_desc)

      y_asc = np.array([grouped_train[str(unique_values_train[i])]['obs'].iloc[0], 
                        grouped_train[str(unique_values_train[i])]['obs'].iloc[1], 
                        grouped_train[str(unique_values_train[i])]['obs'].iloc[2], 
                        grouped_train[str(unique_values_train[i])]['obs'].iloc[3], 
                        grouped_train[str(unique_values_train[i])]['obs'].iloc[4]])
      
      y_train_asc.append(y_asc)

      y_desc = np.array([grouped_train[str(unique_values_train[i])]['obs'].iloc[4], 
                         grouped_train[str(unique_values_train[i])]['obs'].iloc[3], 
                         grouped_train[str(unique_values_train[i])]['obs'].iloc[2], 
                         grouped_train[str(unique_values_train[i])]['obs'].iloc[1], 
                         grouped_train[str(unique_values_train[i])]['obs'].iloc[0]])
      
      y_train_desc.append(y_desc)



  for i in range(0,len(grouped_test)):

    if len(grouped_test[str(unique_values_test[i])]) == 5:
    
      x_asc = np.array([grouped_test[str(unique_values_test[i])]['ens_mean'].iloc[0], 
                        grouped_test[str(unique_values_test[i])]['ens_mean'].iloc[1], 
                        grouped_test[str(unique_values_test[i])]['ens_mean'].iloc[2], 
                        grouped_test[str(unique_values_test[i])]['ens_mean'].iloc[3], 
                        grouped_test[str(unique_values_test[i])]['ens_mean'].iloc[4]])
      
      X_test_asc.append(x_asc)

      x_desc = np.array([grouped_test[str(unique_values_test[i])]['ens_mean'].iloc[4], 
                         grouped_test[str(unique_values_test[i])]['ens_mean'].iloc[3], 
                         grouped_test[str(unique_values_test[i])]['ens_mean'].iloc[2], 
                         grouped_test[str(unique_values_test[i])]['ens_mean'].iloc[1], 
                         grouped_test[str(unique_values_test[i])]['ens_mean'].iloc[0]])

      X_test_desc.append(x_desc)

      y_asc = np.array([grouped_test[str(unique_values_test[i])]['obs'].iloc[0], 
                        grouped_test[str(unique_values_test[i])]['obs'].iloc[1], 
                        grouped_test[str(unique_values_test[i])]['obs'].iloc[2], 
                        grouped_test[str(unique_values_test[i])]['obs'].iloc[3], 
                        grouped_test[str(unique_values_test[i])]['obs'].iloc[4]])
          
      y_test_asc.append(y_asc)

      y_desc = np.array([grouped_test[str(unique_values_test[i])]['obs'].iloc[4], 
                         grouped_test[str(unique_values_test[i])]['obs'].iloc[3], 
                         grouped_test[str(unique_values_test[i])]['obs'].iloc[2], 
                         grouped_test[str(unique_values_test[i])]['obs'].iloc[1], 
                         grouped_test[str(unique_values_test[i])]['obs'].iloc[0]])
          
      y_test_desc.append(y_desc)

      X_asc_var = np.array([grouped_test[str(unique_values_test[i])]['ens_var'].iloc[0], 
                        grouped_test[str(unique_values_test[i])]['ens_var'].iloc[1], 
                        grouped_test[str(unique_values_test[i])]['ens_var'].iloc[2], 
                        grouped_test[str(unique_values_test[i])]['ens_var'].iloc[3], 
                        grouped_test[str(unique_values_test[i])]['ens_var'].iloc[4]])
          
      X_test_asc_var.append(X_asc_var)

      X_desc_var = np.array([grouped_test[str(unique_values_test[i])]['ens_var'].iloc[4], 
                         grouped_test[str(unique_values_test[i])]['ens_var'].iloc[3], 
                         grouped_test[str(unique_values_test[i])]['ens_var'].iloc[2], 
                         grouped_test[str(unique_values_test[i])]['ens_var'].iloc[1], 
                         grouped_test[str(unique_values_test[i])]['ens_var'].iloc[0]])
          
      X_test_desc_var.append(X_desc_var)


  return X_train_asc, y_train_asc, X_train_desc, y_train_desc, X_test_asc, y_test_asc, X_test_desc, y_test_desc, X_test_asc_var, X_test_desc_var


def numpy_to_torch(X_train, y_train, X_test, y_test, Transpose_X = False, Transpose_y = False):

  Xtrain, Ytrain, Xtest, Ytest = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

  if Transpose_X == True:

    Xtrain = Xtrain.T
    Xtest = Xtest.T

  if Transpose_y == True:

    Ytrain = Ytrain.T
    Ytest = Ytest.T

  X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
  Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

  X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
  Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)
  
  return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch


class LSTM(nn.Module):

    def __init__(self, output_size, input_size, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        
        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size = hidden_size,
                            num_layers=num_layers, dropout = dropout)
      
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

      # Initialize Hidden States
      h_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
      c_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size))
      
      # Propagate input through LSTM
      out, (hn, cn) = self.lstm(x, (h_0, c_0))

      #h_out = h_out[self.num_layers-1].view(-1, self.hidden_size)
      
      out = self.fc(out[:, :, :])
      
      return out



def LSTM_Simple(X_train, y_train, X_test, hidden_size, num_layers, num_epochs, batch_size, learning_rate, dropout):

  input_size = 5
  output_size = 5

  hidden_size = hidden_size
  num_layers = num_layers

  dropout = dropout

  lstm_model = LSTM(output_size, input_size, hidden_size, num_layers, dropout)

  # Initialize array of losses 
  losses = np.full(num_epochs, np.nan)

  # Decide Criterion and Optimizer
  criterion = torch.nn.MSELoss()  
  optimizer = torch.optim.Adam(lstm_model.parameters(), lr = learning_rate)

  # Train model

  n_batches = int(X_train.shape[1] / batch_size)

  with trange(num_epochs) as tr:
    for it in tr:
        
      batch_loss = 0.

      for b in range(n_batches):

          input_batch = X_train[:, b: b + batch_size, :]
          target_batch = y_train[:, b: b + batch_size, :]

          # Initialize Output Tensor
          outputs = torch.zeros(1, batch_size, target_batch.shape[2])

          # Run Model
          outputs = lstm_model(input_batch)

          # Zero the gradient
          optimizer.zero_grad()

          # Calculate Loss
          loss = criterion(outputs, target_batch)
          batch_loss = batch_loss + loss.item()
          
          # Backpropagation
          loss.backward()
          optimizer.step()

      # Epoch Loss
      batch_loss = batch_loss/ n_batches 
      losses[it] = batch_loss
      
      # progress bar 
      tr.set_postfix(loss="{0:.3f}".format(batch_loss))

  lstm_model.eval()
  predict = lstm_model(X_test)

  return predict.detach().numpy(), losses



class lstm_encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)


    def forward(self, x_input):

        h_0 = torch.zeros(self.num_layers, x_input.size(1), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x_input.size(1), self.hidden_size)
        
        lstm_out, (hn, cn) = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size), (h_0, c_0))

        return lstm_out, (hn, cn)    
    
class lstm_decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers):

        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
        self.linear = nn.Linear(hidden_size, output_size)           

    def forward(self, x_input, encoder_hidden_states):

        x_input = x_input.unsqueeze(1)

        out, (hn, cn) = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size), encoder_hidden_states)
        out = self.linear(out[:, -1, :])     
        
        return out, (hn, cn)

class lstm_seq2seq(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers):

        super(lstm_seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size, output_size = output_size, num_layers = num_layers)

    def train_model(self, 
                    input_tensor, 
                    target_tensor, 
                    n_epochs, 
                    target_len, 
                    batch_size, 
                    teacher_forcing_ratio, 
                    learning_rate):
        
        
        # Initialize array of losses 
        losses = np.full(n_epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss()

        n_batches = int(input_tensor.shape[1] / batch_size)

        with trange(n_epochs) as tr:

          for it in tr:
            
            batch_loss = 0.

            # Shuffle to stabilize learning
            idx = torch.randperm(input_tensor.shape[1])
            input = input_tensor[:,idx].view(input_tensor.size())
            target = target_tensor[:,idx].view(target_tensor.size())

            for b in range(n_batches):
   
                input_batch = input[:, b: b + batch_size, :]
                target_batch = target[:, b: b + batch_size, :]

                # Initialize Output Tensor
                outputs = torch.zeros(target_len, batch_size, target_batch.shape[2])
                
                # Zero the gradient
                optimizer.zero_grad()

                for ba in range(0, batch_size):

                  # Run Encoder
                  encoder_output, encoder_hidden = self.encoder(input_batch[:, ba, :])

                  decoder_input = input_batch[-1, :, :]   
                  decoder_hidden = encoder_hidden

                  decoder_output, decoder_hidden = self.decoder(decoder_input[ba,:], (decoder_hidden[0][:,0,:].unsqueeze(1), decoder_hidden[1][:,0,:].unsqueeze(1)))
                  outputs[0][ba] = decoder_output
                  decoder_input = decoder_output
                  
                  for t in range(1, target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                  
                    outputs[t][ba] = decoder_output
                    
                    # Teacher forcing
                    if random.random() < teacher_forcing_ratio:
                        decoder_input = target_batch[t, ba, :]
                    
                    # Recursively 
                    else:
                      decoder_input = decoder_output

                # Calculate Loss 
                loss = criterion(outputs, target_batch)
                batch_loss = batch_loss + loss.item()
                
                # Backpropagation
                loss.backward()
                optimizer.step()

            # loss for epoch 
            batch_loss = batch_loss / n_batches 
            losses[it] = batch_loss

            # progress bar 
            tr.set_postfix(loss="{0:.3f}".format(batch_loss))
                    
        return losses

    def predict(self, input_tensor, target_len, output_size):

        # Encode input_tensor
        input_tensor = input_tensor    
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # Initialize tensor for predictions
        outputs = torch.zeros(target_len, output_size)

        # Decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden

        for t in range(target_len): 

          decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
          outputs[t] = decoder_output
          decoder_input = decoder_output
            
        np_outputs = outputs.detach().numpy()
        
        return np_outputs

def Train_all_NNs(X_train, X_test, y_train, y_test, name, teacher_forcing_ratio_1, teacher_forcing_ratio_2, learning_rate_1, learning_rate_2):

  seed = 1234
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.enabled = False

  X_train_asc, y_train_asc, X_train_desc, y_train_desc, X_test_asc, y_test_asc, X_test_desc, y_test_desc, X_test_asc_var, X_test_desc_var = PrepareforNets(X_train, X_test, y_train, y_test)

  X_train_torch, Y_train_torch, X_test_torch, Y_test_torch = numpy_to_torch(X_train_asc, y_train_asc, X_test_asc, y_test_asc, Transpose_X = False, Transpose_y = False)

  X_train_torch_1F, Y_train_torch_1F, X_test_torch_1F, Y_test_torch_1F = numpy_to_torch(X_train_asc, y_train_asc, X_test_asc, y_test_asc, Transpose_X = True, Transpose_y = True)
  X_train_torch_5F, Y_train_torch_5F, X_test_torch_5F, Y_test_torch_5F = numpy_to_torch(X_train_asc, y_train_asc, X_test_asc, y_test_asc, Transpose_X = True, Transpose_y = False)

  X_train_torch_1F_d, Y_train_torch_1F_d, X_test_torch_1F_d, Y_test_torch_1F_d = numpy_to_torch(X_train_desc, y_train_asc, X_test_desc, y_test_asc, Transpose_X = True, Transpose_y = True)
  X_train_torch_5F_d, Y_train_torch_5F_d, X_test_torch_5F_d, Y_test_torch_5F_d = numpy_to_torch(X_train_desc, y_train_asc, X_test_desc, y_test_asc, Transpose_X = True, Transpose_y = False)

  y_TRUE = pd.DataFrame(y_test_asc).T
  y_TRUE_36 = pd.DataFrame(y_TRUE.iloc[0,:])
  y_TRUE_48 = pd.DataFrame(y_TRUE.iloc[1,:])
  y_TRUE_60 = pd.DataFrame(y_TRUE.iloc[2,:])
  y_TRUE_72 = pd.DataFrame(y_TRUE.iloc[3,:])
  y_TRUE_84 = pd.DataFrame(y_TRUE.iloc[4,:])

  result_lstm_simple, loss_simple = LSTM_Simple( X_train       = X_train_torch.unsqueeze(0), 
                                  y_train       = Y_train_torch.unsqueeze(0), 
                                  X_test        = X_test_torch.unsqueeze(0), 
                                  hidden_size   = 25, 
                                  num_layers    = 2, 
                                  num_epochs    = 250, 
                                  batch_size    = 250, 
                                  learning_rate = 0.005, 
                                  dropout       = 0.2)

  LSTM_Simple_result = {}
  LSTM_Simple_result_quantiles = {}

  for i in range(5):

    LSTM_Simple_result[i] = pd.DataFrame(result_lstm_simple.squeeze()).iloc[:,i]

    results_975 = []
    results_75 = []
    results_5 = []
    results_25 = []
    results_025 = []

    for j in range(len(LSTM_Simple_result[i])):
      results = np.random.normal(LSTM_Simple_result[i][j], X_test_asc_var[j][i], 10000)

      results_975.append(pd.DataFrame(results).quantile(.975, axis=0))
      results_75.append(pd.DataFrame(results).quantile(.75, axis=0))
      results_5.append(pd.DataFrame(results).quantile(.5, axis=0))
      results_25.append(pd.DataFrame(results).quantile(.25, axis=0))
      results_025.append(pd.DataFrame(results).quantile(.025, axis=0))

    LSTM_Simple_result_quantiles[i] =pd.concat([pd.DataFrame(results_025).reset_index(drop=True), 
                                                pd.DataFrame(results_25).reset_index(drop=True), 
                                                pd.DataFrame(results_5).reset_index(drop=True), 
                                                pd.DataFrame(results_75).reset_index(drop=True), 
                                                pd.DataFrame(results_975).reset_index(drop=True)],
                                                axis = 1)

  results_s36 = Backtesting_per_Hour(LSTM_Simple_result_quantiles[0], y_TRUE_36.iloc[:,0], '36')
  results_s48 = Backtesting_per_Hour(LSTM_Simple_result_quantiles[1], y_TRUE_48.iloc[:,0], '48')
  results_s60 = Backtesting_per_Hour(LSTM_Simple_result_quantiles[2], y_TRUE_60.iloc[:,0], '60')
  results_s72 = Backtesting_per_Hour(LSTM_Simple_result_quantiles[3], y_TRUE_72.iloc[:,0], '72')
  results_s84 = Backtesting_per_Hour(LSTM_Simple_result_quantiles[4], y_TRUE_84.iloc[:,0], '84')

  result_lstm_simple = pd.concat([pd.DataFrame(results_s36.values, columns = ['36'], index = results_s36.index), 
                                  pd.DataFrame(results_s48.values, columns = ['48'], index = results_s36.index), 
                                  pd.DataFrame(results_s60.values, columns = ['60'], index = results_s36.index), 
                                  pd.DataFrame(results_s72.values, columns = ['72'], index = results_s36.index), 
                                  pd.DataFrame(results_s84.values, columns = ['84'], index = results_s36.index)], axis = 1)

  model_5F  = lstm_seq2seq( input_size   = 1, 
                          hidden_size  = 25, 
                          output_size  = 5, 
                          num_layers   = 1)

  loss_5F = model_5F.train_model( input_tensor          = X_train_torch_5F.unsqueeze(2), 
                                  target_tensor         = Y_train_torch_5F.unsqueeze(0), 
                                  n_epochs              = 250, 
                                  target_len            = 1, 
                                  batch_size            = 250, 
                                  teacher_forcing_ratio = teacher_forcing_ratio_1, 
                                  learning_rate         = learning_rate_1)


  LSTM_5F_result = {}
  LSTM_5F_result_quantiles = {}

  seq_res       = model_5F.predict(X_test_torch_5F[:,0].unsqueeze(1).unsqueeze(2), target_len = 1, output_size = 5)
  seq_res_df    = pd.DataFrame(seq_res, index = [str(0)])

  for i in range(1,len(y_TRUE_36)):
    seq_res     = model_5F.predict(X_test_torch_5F[:,i].unsqueeze(1).unsqueeze(2), target_len = 1, output_size = 5)
    seq_res_df  = pd.concat([seq_res_df, pd.DataFrame(seq_res, index = [str(i)])], axis = 0)

  for i in range(5):

    #LSTM_5F_result[i] = pd.DataFrame(res.squeeze()).iloc[:,i]

    results_975 = []
    results_75 = []
    results_5 = []
    results_25 = []
    results_025 = []

    for j in range(len(seq_res_df)):
      results = np.random.normal(seq_res_df.iloc[j,i], X_test_asc_var[j][i], 10000)

      results_975.append(pd.DataFrame(results).quantile(.975, axis=0))
      results_75.append(pd.DataFrame(results).quantile(.75, axis=0))
      results_5.append(pd.DataFrame(results).quantile(.5, axis=0))
      results_25.append(pd.DataFrame(results).quantile(.25, axis=0))
      results_025.append(pd.DataFrame(results).quantile(.025, axis=0))

    LSTM_5F_result_quantiles[i] =pd.concat([pd.DataFrame(results_025).reset_index(drop=True), 
                                                pd.DataFrame(results_25).reset_index(drop=True), 
                                                pd.DataFrame(results_5).reset_index(drop=True), 
                                                pd.DataFrame(results_75).reset_index(drop=True), 
                                                pd.DataFrame(results_975).reset_index(drop=True)],
                                                axis = 1)

  results_s36 = Backtesting_per_Hour(LSTM_5F_result_quantiles[0], y_TRUE_36.iloc[:,0], '36')
  results_s48 = Backtesting_per_Hour(LSTM_5F_result_quantiles[1], y_TRUE_48.iloc[:,0], '48')
  results_s60 = Backtesting_per_Hour(LSTM_5F_result_quantiles[2], y_TRUE_60.iloc[:,0], '60')
  results_s72 = Backtesting_per_Hour(LSTM_5F_result_quantiles[3], y_TRUE_72.iloc[:,0], '72')
  results_s84 = Backtesting_per_Hour(LSTM_5F_result_quantiles[4], y_TRUE_84.iloc[:,0], '84')

  result_lstm_5F = pd.concat([pd.DataFrame(results_s36.values, columns = ['36'], index = results_s36.index), 
                                  pd.DataFrame(results_s48.values, columns = ['48'], index = results_s36.index), 
                                  pd.DataFrame(results_s60.values, columns = ['60'], index = results_s36.index), 
                                  pd.DataFrame(results_s72.values, columns = ['72'], index = results_s36.index), 
                                  pd.DataFrame(results_s84.values, columns = ['84'], index = results_s36.index)], axis = 1)

  model_1F  = lstm_seq2seq( input_size   = 1, 
                          hidden_size  = 25, 
                          output_size  = 1, 
                          num_layers   = 1)

  loss_1F = model_1F.train_model( input_tensor          = X_train_torch_1F.unsqueeze(2), 
                                target_tensor         = Y_train_torch_1F.unsqueeze(2), 
                                n_epochs              = 250, 
                                target_len            = 5, 
                                batch_size            = 100, 
                                teacher_forcing_ratio = teacher_forcing_ratio_2, 
                                learning_rate         = learning_rate_2)

  LSTM_1F_result_quantiles = {}

  seq_res       = model_1F.predict(X_test_torch_1F[:,0].unsqueeze(1).unsqueeze(2), target_len = 5, output_size = 1)
  seq_res_df    = pd.DataFrame(seq_res.T, index = [str(0)])

  for i in range(1,len(y_TRUE_36)):
    seq_res     = model_1F.predict(X_test_torch_1F[:,i].unsqueeze(1).unsqueeze(2), target_len = 5, output_size = 1)
    seq_res_df  = pd.concat([seq_res_df, pd.DataFrame(seq_res.T, index = [str(i)])], axis = 0)

  for i in range(5):

    #LSTM_5F_result[i] = pd.DataFrame(res.squeeze()).iloc[:,i]

    results_975 = []
    results_75 = []
    results_5 = []
    results_25 = []
    results_025 = []

    for j in range(len(seq_res_df)):
      results = np.random.normal(seq_res_df.iloc[j,i], X_test_asc_var[j][i], 10000)

      results_975.append(pd.DataFrame(results).quantile(.975, axis=0))
      results_75.append(pd.DataFrame(results).quantile(.75, axis=0))
      results_5.append(pd.DataFrame(results).quantile(.5, axis=0))
      results_25.append(pd.DataFrame(results).quantile(.25, axis=0))
      results_025.append(pd.DataFrame(results).quantile(.025, axis=0))

    LSTM_1F_result_quantiles[i] =pd.concat([pd.DataFrame(results_025).reset_index(drop=True), 
                                                pd.DataFrame(results_25).reset_index(drop=True), 
                                                pd.DataFrame(results_5).reset_index(drop=True), 
                                                pd.DataFrame(results_75).reset_index(drop=True), 
                                                pd.DataFrame(results_975).reset_index(drop=True)],
                                                axis = 1)

  results_s36 = Backtesting_per_Hour(LSTM_1F_result_quantiles[0], y_TRUE_36.iloc[:,0], '36')
  results_s48 = Backtesting_per_Hour(LSTM_1F_result_quantiles[1], y_TRUE_48.iloc[:,0], '48')
  results_s60 = Backtesting_per_Hour(LSTM_1F_result_quantiles[2], y_TRUE_60.iloc[:,0], '60')
  results_s72 = Backtesting_per_Hour(LSTM_1F_result_quantiles[3], y_TRUE_72.iloc[:,0], '72')
  results_s84 = Backtesting_per_Hour(LSTM_1F_result_quantiles[4], y_TRUE_84.iloc[:,0], '84')

  result_lstm_1F = pd.concat([pd.DataFrame(results_s36.values, columns = ['36'], index = results_s36.index), 
                                  pd.DataFrame(results_s48.values, columns = ['48'], index = results_s36.index), 
                                  pd.DataFrame(results_s60.values, columns = ['60'], index = results_s36.index), 
                                  pd.DataFrame(results_s72.values, columns = ['72'], index = results_s36.index), 
                                  pd.DataFrame(results_s84.values, columns = ['84'], index = results_s36.index)], axis = 1)



  model_5F_d  = lstm_seq2seq( input_size   = 1, 
                          hidden_size  = 25, 
                          output_size  = 5, 
                          num_layers   = 1)

  loss_5F_d = model_5F_d.train_model( input_tensor          = X_train_torch_5F_d.unsqueeze(2), 
                                target_tensor         = Y_train_torch_5F_d.unsqueeze(0), 
                                n_epochs              = 250, 
                                target_len            = 1, 
                                batch_size            = 250, 
                                teacher_forcing_ratio = teacher_forcing_ratio_1, 
                                learning_rate         = learning_rate_1)



  LSTM_5F_d_result_quantiles = {}

  seq_res       = model_5F_d.predict(X_test_torch_5F_d[:,0].unsqueeze(1).unsqueeze(2), target_len = 1, output_size = 5)
  seq_res_df    = pd.DataFrame(seq_res, index = [str(0)])

  for i in range(1,len(y_TRUE_36)):
    seq_res     = model_5F_d.predict(X_test_torch_5F_d[:,i].unsqueeze(1).unsqueeze(2), target_len = 1, output_size = 5)
    seq_res_df  = pd.concat([seq_res_df, pd.DataFrame(seq_res, index = [str(i)])], axis = 0)

  for i in range(5):

    #LSTM_5F_result[i] = pd.DataFrame(res.squeeze()).iloc[:,i]

    results_975 = []
    results_75 = []
    results_5 = []
    results_25 = []
    results_025 = []

    for j in range(len(seq_res_df)):
      results = np.random.normal(seq_res_df.iloc[j,i], X_test_desc_var[j][i], 10000)

      results_975.append(pd.DataFrame(results).quantile(.975, axis=0))
      results_75.append(pd.DataFrame(results).quantile(.75, axis=0))
      results_5.append(pd.DataFrame(results).quantile(.5, axis=0))
      results_25.append(pd.DataFrame(results).quantile(.25, axis=0))
      results_025.append(pd.DataFrame(results).quantile(.025, axis=0))

    LSTM_5F_d_result_quantiles[i] =pd.concat([pd.DataFrame(results_025).reset_index(drop=True), 
                                                pd.DataFrame(results_25).reset_index(drop=True), 
                                                pd.DataFrame(results_5).reset_index(drop=True), 
                                                pd.DataFrame(results_75).reset_index(drop=True), 
                                                pd.DataFrame(results_975).reset_index(drop=True)],
                                                axis = 1)

  results_s36 = Backtesting_per_Hour(LSTM_5F_d_result_quantiles[0], y_TRUE_36.iloc[:,0], '36')
  results_s48 = Backtesting_per_Hour(LSTM_5F_d_result_quantiles[1], y_TRUE_48.iloc[:,0], '48')
  results_s60 = Backtesting_per_Hour(LSTM_5F_d_result_quantiles[2], y_TRUE_60.iloc[:,0], '60')
  results_s72 = Backtesting_per_Hour(LSTM_5F_d_result_quantiles[3], y_TRUE_72.iloc[:,0], '72')
  results_s84 = Backtesting_per_Hour(LSTM_5F_d_result_quantiles[4], y_TRUE_84.iloc[:,0], '84')

  result_lstm_5F_d = pd.concat([pd.DataFrame(results_s36.values, columns = ['36'], index = results_s36.index), 
                                  pd.DataFrame(results_s48.values, columns = ['48'], index = results_s36.index), 
                                  pd.DataFrame(results_s60.values, columns = ['60'], index = results_s36.index), 
                                  pd.DataFrame(results_s72.values, columns = ['72'], index = results_s36.index), 
                                  pd.DataFrame(results_s84.values, columns = ['84'], index = results_s36.index)], axis = 1)


  model_1F_d  = lstm_seq2seq( input_size   = 1, 
                          hidden_size  = 25, 
                          output_size  = 1, 
                          num_layers   = 1)

  loss_1F_d = model_1F_d.train_model( input_tensor          = X_train_torch_1F_d.unsqueeze(2), 
                                target_tensor         = Y_train_torch_1F_d.unsqueeze(2), 
                                n_epochs              = 250, 
                                target_len            = 5, 
                                batch_size            = 100, 
                                teacher_forcing_ratio = teacher_forcing_ratio_2, 
                                learning_rate         = learning_rate_2)

  LSTM_1F_d_result_quantiles = {}

  seq_res       = model_1F_d.predict(X_test_torch_1F_d[:,0].unsqueeze(1).unsqueeze(2), target_len = 5, output_size = 1)
  seq_res_df    = pd.DataFrame(seq_res.T, index = [str(0)])

  for i in range(1,len(y_TRUE_36)):
    seq_res     = model_1F_d.predict(X_test_torch_1F_d[:,i].unsqueeze(1).unsqueeze(2), target_len = 5, output_size = 1)
    seq_res_df  = pd.concat([seq_res_df, pd.DataFrame(seq_res.T, index = [str(i)])], axis = 0)

  for i in range(5):

    #LSTM_5F_result[i] = pd.DataFrame(res.squeeze()).iloc[:,i]

    results_975 = []
    results_75 = []
    results_5 = []
    results_25 = []
    results_025 = []

    for j in range(len(seq_res_df)):
      results = np.random.normal(seq_res_df.iloc[j,i], X_test_desc_var[j][i], 10000)

      results_975.append(pd.DataFrame(results).quantile(.975, axis=0))
      results_75.append(pd.DataFrame(results).quantile(.75, axis=0))
      results_5.append(pd.DataFrame(results).quantile(.5, axis=0))
      results_25.append(pd.DataFrame(results).quantile(.25, axis=0))
      results_025.append(pd.DataFrame(results).quantile(.025, axis=0))

    LSTM_1F_d_result_quantiles[i] =pd.concat([pd.DataFrame(results_025).reset_index(drop=True), 
                                                pd.DataFrame(results_25).reset_index(drop=True), 
                                                pd.DataFrame(results_5).reset_index(drop=True), 
                                                pd.DataFrame(results_75).reset_index(drop=True), 
                                                pd.DataFrame(results_975).reset_index(drop=True)],
                                                axis = 1)

  results_s36 = Backtesting_per_Hour(LSTM_1F_d_result_quantiles[0], y_TRUE_36.iloc[:,0], '36')
  results_s48 = Backtesting_per_Hour(LSTM_1F_d_result_quantiles[1], y_TRUE_48.iloc[:,0], '48')
  results_s60 = Backtesting_per_Hour(LSTM_1F_d_result_quantiles[2], y_TRUE_60.iloc[:,0], '60')
  results_s72 = Backtesting_per_Hour(LSTM_1F_d_result_quantiles[3], y_TRUE_72.iloc[:,0], '72')
  results_s84 = Backtesting_per_Hour(LSTM_1F_d_result_quantiles[4], y_TRUE_84.iloc[:,0], '84')

  result_lstm_1F_d = pd.concat([pd.DataFrame(results_s36.values, columns = ['36'], index = results_s36.index), 
                                  pd.DataFrame(results_s48.values, columns = ['48'], index = results_s36.index), 
                                  pd.DataFrame(results_s60.values, columns = ['60'], index = results_s36.index), 
                                  pd.DataFrame(results_s72.values, columns = ['72'], index = results_s36.index), 
                                  pd.DataFrame(results_s84.values, columns = ['84'], index = results_s36.index)], axis = 1)

 
  result_lstm_simple['Average'] = result_lstm_simple.mean(axis=1)
  result_lstm_5F['Average']     = result_lstm_5F.mean(axis=1)
  result_lstm_1F['Average']     = result_lstm_1F.mean(axis=1)
  result_lstm_5F_d['Average']   = result_lstm_5F_d.mean(axis=1)
  result_lstm_1F_d['Average']   = result_lstm_1F_d.mean(axis=1)

  RESULTS = pd.concat([result_lstm_simple, result_lstm_5F, result_lstm_1F, result_lstm_5F_d, result_lstm_1F_d], keys = ['LSTM Simple', 'Seq-to-Features', 'Seq-to-Seq', 'Seq-to-Features reversed', 'Seq-to-Seq reversed'], axis = 1)

  RESULTS.to_excel('Results NN ' + name + '.xlsx', index = True)

  return loss_simple, loss_5F, loss_1F, loss_5F_d, loss_1F_d

