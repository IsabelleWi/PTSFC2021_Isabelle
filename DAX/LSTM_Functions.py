#Imports
import pandas as pd
import numpy as np

import random
import torch
import torch.nn as nn
from torch.autograd import Variable

from Preprocessing_Evaluation_Functions import *

def MakeTorch(X_train, y_train, X_test, y_test):
  
  x11 = np.array(get_technical_indicators(X_train['x-5']))
  x12 = np.array(get_technical_indicators(X_train['x-4']))
  x13 = np.array(get_technical_indicators(X_train['x-3']))
  x14 = np.array(get_technical_indicators(X_train['x-2']))
  x15 = np.array(get_technical_indicators(X_train['x-1']))


  x31= np.array(get_technical_indicators(pd.concat([X_train['x-5'][-17:],X_test['x-5']])))
  x32= np.array(get_technical_indicators(pd.concat([X_train['x-4'][-17:],X_test['x-4']])))
  x33= np.array(get_technical_indicators(pd.concat([X_train['x-3'][-17:],X_test['x-3']])))
  x34= np.array(get_technical_indicators(pd.concat([X_train['x-2'][-17:],X_test['x-2']])))
  x35= np.array(get_technical_indicators(pd.concat([X_train['x-1'][-17:],X_test['x-1']])))

  return Variable(torch.Tensor(np.stack((x11[15:],x12[15:],x13[15:],x14[15:],x15[15:]), axis = 1))), Variable(torch.Tensor(np.array(y_train)[15:])), Variable(torch.Tensor(np.stack((x31[15:],x32[15:],x33[15:],x34[15:],x35[15:]), axis = 1))), Variable(torch.Tensor(np.array(y_test)[15:]))

def MakeTorch_for_prediction(X_train, y_train, X_test, y_test):
  
  x11 = np.array(get_technical_indicators(X_train['x-5']))
  x12 = np.array(get_technical_indicators(X_train['x-4']))
  x13 = np.array(get_technical_indicators(X_train['x-3']))
  x14 = np.array(get_technical_indicators(X_train['x-2']))
  x15 = np.array(get_technical_indicators(X_train['x-1']))


  x31= np.array(get_technical_indicators(pd.concat([X_train['x-5'][-17:],X_test['x-5']])))
  x32= np.array(get_technical_indicators(pd.concat([X_train['x-4'][-17:],X_test['x-4']])))
  x33= np.array(get_technical_indicators(pd.concat([X_train['x-3'][-17:],X_test['x-3']])))
  x34= np.array(get_technical_indicators(pd.concat([X_train['x-2'][-17:],X_test['x-2']])))
  x35= np.array(get_technical_indicators(pd.concat([X_train['x-1'][-17:],X_test['x-1']])))

  return Variable(torch.Tensor(np.stack((x11[15:],x12[15:],x13[15:],x14[15:],x15[15:]), axis = 1))), Variable(torch.Tensor(np.array(y_train)[15:])), Variable(torch.Tensor(np.stack((x31[17:],x32[17:],x33[17:],x34[17:],x35[17:]), axis = 1))), Variable(torch.Tensor(np.array(y_test)))


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout = 0.1)
      
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        out, (hn, cn) = self.lstm(x, (h_0, c_0))

        #h_out = h_out[self.num_layers-1].view(-1, self.hidden_size)
        
        out = self.fc(out[:, -1, :])
        
        
        return out


def Train_LSTM(x_train, y_train, x_test, num_epochs = 1000, learning_rate = 0.1):

  seed = 0
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.enabled = False
  torch.use_deterministic_algorithms(True)

  input_size = 5
  hidden_size = 25
  num_layers = 2
  seq_length = 5

  num_classes = 1

  lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)

  criterion = torch.nn.L1Loss() 
  optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

  for epoch in range(num_epochs):
      outputs = lstm(x_train)
      optimizer.zero_grad()

      loss = criterion(outputs, y_train)
      
      loss.backward()
      
      optimizer.step()
      if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

  lstm.eval()
  train_predict = lstm(x_test)
  return pd.DataFrame(train_predict.detach().numpy()).T