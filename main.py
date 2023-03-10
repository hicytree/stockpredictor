# Make sure that you have all these libaries available to run the code successfully
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
import DataGenerator

#DATA ACQUISITION####################################################################################################
data_source = 'kaggle'

if data_source == "alphavantage":
    api_key = '4IKUPL5P8ZAMBM1X'
    ticker = "AAL"
    
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
    file_to_save = 'stock_market_data-%s.csv'%ticker
    
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            data = data['Weekly Time Series']
            df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
            for k, v in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(), float(v['3. low']), float(v['2. high']), float(v['4. close']), float(v['1. open'])]
                df.loc[-1,:] = data_row
                df.index = df.index + 1
            df.to_csv(file_to_save, index=False)
            print('Loaded data from Alpha Vantage')
            print('Data saved to : %s'%file_to_save)
            
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)  
else:
    df = pd.read_csv(os.path.join('Stocks', 'hpq.us.txt'), delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    print('Loaded data from Kaggle repository')

#DATA VISUALIZATION####################################################################################################
# df = df.sort_values('Date')
# print(df.head())

# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
# plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
# plt.xlabel('Date',fontsize=18)
# plt.ylabel('Mid Price',fontsize=18)
# plt.show()

#SETTING UP THE DATA###################################################################################################
high_prices = df.loc[:,'High']
low_prices = df.loc[:,'Low']
mid_prices = (high_prices + low_prices)/2.0

train_data = mid_prices[:11000]
test_data = mid_prices[11000:]

scaler = MinMaxScaler()
train_data = train_data.values.reshape(-1, 1)
test_data = test_data.values.reshape(-1, 1)

smoothing_window_size = 2500
for di in range(0, 10000, smoothing_window_size):
    scaler.fit(train_data[di: di + smoothing_window_size, :])
    train_data[di: di + smoothing_window_size, :] = scaler.transform(train_data[di: di + smoothing_window_size, :])
    
scaler.fit(train_data[di + smoothing_window_size:, :])
train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])

train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)

EMA = 0.0
gamma = 0.1
for ti in range(11000):
  EMA = gamma * train_data[ti] + (1 - gamma) * EMA
  train_data[ti] = EMA

all_mid_data = np.concatenate([train_data, test_data], axis=0)

#NAIVE STANDARD AVERAGING ALGORITHM#####################################################################################
# window_size = 100
# N = train_data.size
# std_avg_predictions = []
# mse_errors = []
# for pred_idx in range(window_size, N):
#     std_avg_predictions.append(np.mean(train_data[pred_idx - window_size: pred_idx]))
#     mse_errors.append((std_avg_predictions[-1] - train_data[pred_idx]) ** 2)
# print('MSE error for standard averaging: %.5f' % (np.mean(mse_errors)))

#DATA VISUALIZATION#####################################################################################################
# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
# plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
# plt.xlabel('Date')
# plt.ylabel('Mid Price')
# plt.legend(fontsize=18)
# plt.show()
        
#EXPONENTIAL MOVING AVERAGE ALGORITHM####################################################################################
# window_size = 100
# N = train_data.size
# run_avg_predictions = []
# mse_errors = []
# running_mean = 0.0
# run_avg_predictions.append(running_mean)
# decay = 0.5

# for pred_idx in range(1, N):
#     running_mean = running_mean * decay + (1.0 - decay) * train_data[pred_idx - 1]
#     run_avg_predictions.append(running_mean)
#     mse_errors.append((run_avg_predictions[-1] - train_data[pred_idx]) ** 2)
# print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))

#DATA VISUALIZATION#####################################################################################################
# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
# plt.plot(range(0,N),run_avg_predictions,color='orange', label='Prediction')
# plt.xlabel('Date')
# plt.ylabel('Mid Price')
# plt.legend(fontsize=18)
# plt.show()

#DATA GENERATOR TEST################################################################################################
# dg = DataGenerator.DataGeneratorSeq(train_data, 5, 5)
# u_data, u_labels = dg.unroll_batches()
# for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):   
#     print('\n\nUnrolled index %d'%ui)
#     dat_ind = dat
#     lbl_ind = lbl
#     print('\tInputs: ',dat )
#     print('\n\tOutput:',lbl)

D = 1
num_unrollings = 50
batch_size = 500
num_nodes = [200, 200, 150]
n_layers = len(num_nodes)
dropout = 0.2

tf.reset_default_graph()

train_inputs, train_outputs = [], []
for ui in range(num_unrollings):
    train_inputs.append(tf.placeholder(tf.float32, shape = [batch_size, D], name='train_inputs_%d'%ui))
    train_outputs.append(tf.placeholder(tf.float32, shape = [batch_size, D], name = 'train_outputs_%d'%ui))

lstm_cells = [tf.contrib.rnn.LSTMCell(num_units = num_nodes[li], state_is_tuple = True, initializer = tf.contrib.layers.xavier_initializer()) for li in range(n_layers)]
drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = 1.0, output_keep_prob = 1.0 - dropout, state_keep_prob = 1.0 - dropout) for lstm in lstm_cells]
drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

w = tf.get_variable('w', shape = [num_nodes[-1], 1], initializer = tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b', initializer = tf.random_uniform([1], -0.1, 0.1))

c, h = [],[]
initial_state = []
for li in range(n_layers):
  c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
  h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
  initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))
  
all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs], axis=0)
all_lstm_outputs, state = tf.nn.dynamic_rnn(drop_multi_cell, all_inputs, initial_state = tuple(initial_state), time_major = True, dtype = tf.float32)
all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size * num_unrollings, num_nodes[-1]])
all_outputs = tf.nn.xw_plus_b(all_lstm_outputs, w, b)
split_outputs = tf.split(all_outputs, num_unrollings, axis=0)