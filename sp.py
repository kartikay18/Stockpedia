import bs4 as bs
import datetime as dt
from matplotlib import style
from predict import predict_stock
import numpy as np
import os
import datetime
from sklearn import preprocessing
import urllib2
import pytz
import pandas as pd
from bs4 import BeautifulSoup
import pickle
import requests
import json
import numpy as np
import pandas as pd
import math, time
import itertools
import keras
import pandas_datareader.data as web
import h5py
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model


stocks = ['VMW','SPLK','GOOG', 'FB', 'NVDA', 'AAPL', 'AMZN', 'MSFT']


def get_stock_data(stock_name, normalize=True):
    #    start = datetime.datetime(1950, 1, 1)
    start = datetime.datetime(2007, 1, 1)
    end = datetime.date.today()
    df = web.DataReader(stock_name, "yahoo", start, end)
    df.drop(['Volume', 'Close'], 1, inplace=True)
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1))
        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1))
        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1))
        df['Adj Close'] = min_max_scaler.fit_transform(df['Adj Close'].values.reshape(-1,1))
    return df


def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix()
    sequence_length = seq_len + 1 # index starting from 0
    result = []
    
    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days
    
    result = np.array(result)


    row = round(0.9 * result.shape[0]) # 90% split
    
    train = result[:int(row), :] # 90% date

    X_train = train[:, :-1] # all data until day m
    y_train = train[:, -1][:,-1] # day m + 1 adjusted close price
    X_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))
    return [X_train, y_train, X_test, y_test]



def build_model2(layers, neurons, d):
    model = Sequential()
    
    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    
    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))
    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))
    # model = load_model('my_LSTM_stock_model1000.h5')
    # adam = keras.optimizers.Adam(decay=0.2)
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    #model.summary()
    return model


def percentage_difference(model, X_test, y_test):
    percentage_diff=[]
    
    p = model.predict(X_test)
    for u in range(len(y_test)): # for each data index in test data
        pr = p[u][0] # pr = prediction on day u
        print pr
    return p

def denormalize(stock_name, normalized_value):
    start = datetime.datetime(2014, 1, 1)
    end = datetime.date.today()
    df = web.DataReader(stock_name, "yahoo", start, end)
    
    df = df['Adj Close'].values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)
    
    #return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new

def plot_result(stock_name, normalized_value_p, normalized_value_y_test):
    newp = denormalize(stock_name, normalized_value_p)
    newy_test = denormalize(stock_name, normalized_value_y_test)


if __name__ == '__main__':
    #initialize()
    symbol="Apple"
    model = load_model('AAPL.h5')
    """    
        for stock in stocks:
        stock_name = stock
        seq_len = 22
        d = 0.2
        shape = [4, seq_len, 1] # feature, window, output
        #neurons = [128, 128, 32, 1]
        neurons=[256,256,32,1]
        epochs = 30
        df__0 = get_stock_data(stock_name, normalize=True)
        X_train, y_train, X_test, y_test = load_data(df__0, seq_len)
        X_train.shape[0], X_train.shape[1], X_train.shape[2]
        y_train.shape[0]
        model = build_model2(shape, neurons, d)
        model.fit(
          X_train,
          y_train,
          batch_size=512,
          epochs=epochs,
          validation_split=0.1,
          verbose=1)
        p = percentage_difference(model, X_test, y_test)
        model.save(stock +'.h5')
    """
        

# add model. predict return the output to result
    url = 'https://data.chastiser11.hasura-app.io/v1/query'

# Sample row input

        

    payload = {"type": "select", "args":{"table": "companyDB", "columns": ["stocksymbol"],
        "where": { "name": symbol} }}
        
    headers = {"content-type": "application/json", "authorization": "Bearer wj7fmf21w6lvu0l4u7vmdef1tqo0cykn"}
        
    r = requests.post(url, json.dumps(payload), headers=headers)
    stock_symbol = json.loads(r.text)

    if r.status_code == 200:
        print 'success'
        predict_stock(stock_symbol[0]['stocksymbol'],10)
    else:
        print 'error: ' + str(r.status_code) + '\n\n' + r.text


#results = p




