# -*- coding: utf-8 -*-
"""
Created on Sat Sep 09 15:47:15 2017

@author: Gayatri

"""

#from keras.layers.core import Dense, Activation, Dropout
#from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.models import Sequential
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn import preprocessing
import requests
import json
import pandas as pd


results = []
results2 = []
period = 10

def predict_stock(stock_name,period):
    
    #    print (model)
    data=pd.read_csv('CSV/'+stock_name+".csv")
    #    print(data.head)
    del data['Date']
    del data['Close']
    del data['Volume']
    #    data.drop('Date')
    #    data.drop('Close')
    #    data.drop('Volume')
    min_max_scaler = preprocessing.MinMaxScaler()
    data['Open'] = min_max_scaler.fit_transform(data.Open.values.reshape(-1,1))
    data['High'] = min_max_scaler.fit_transform(data.High.values.reshape(-1,1))
    data['Low'] = min_max_scaler.fit_transform(data.Low.values.reshape(-1,1))
    data['Adj Close'] = min_max_scaler.fit_transform(data['Adj Close'].values.reshape(-1,1))
    #    print(data.head)
    amount_of_features = len(data.columns)
    data = data.as_matrix()
    sequence_length=22
    result=[]
    #    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
    result.append(data[0: 0+ sequence_length]) # index : index + 22days
    
    result = np.array(result)
    #    print(result[0])
    X_test=result
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))
    #    print(X_test.shape)
    op=[]
    model_open=load_model(stock_name+'_open.h5')
    model_high=load_model(stock_name+'_high.h5')
    model_low=load_model(stock_name+'_low.h5')
    model_close=load_model(stock_name+'_close.h5')
    
    open=[]
    high=[]
    low=[]
    close=[]
    for i in range(period):
        p_open= model_open.predict(X_test)[0][0]
        open.append(p_open)
        p_high= model_high.predict(X_test)[0][0]
        high.append(p_high)
        p_low= model_low.predict(X_test)[0][0]
        low.append(p_low)
        p_close= model_close.predict(X_test)[0][0]
        close.append(p_close)
        p=[p_open,p_high,p_low,p_close]
        #        print (p)
        #        print X_test[0][0]
        #    print X_test[0][1]
        
        temp=X_test[0][1:22]
        #        print temp.shape
        
        X_test=X_test[0][1:22]
        #        print  "shape of xtest", X_test.shape
        #        print p
        #    p.reshape(1,4)
        #        print "shape of p",len(p)
        my_list=[]
        my_list.extend(p)
        #print list_array.shape
        #    np.append(X_test,p,axis=0)
        #my_list.reshape(1,4)
        X_test=np.concatenate((X_test,[my_list]))
        X_test=np.array([X_test])
    #        print X_test.shape
    print open
    #period = len(open)
    print period
    print high
    print low
    print close
    
    open2 = pd.Series(open).to_json(orient='values')
    #print type(open2)
    #print type(open2[0])
    
    #payload = {"type": "insert", "args":{"table": "stockTest", "objects": [{"stocksymbol": stock_name, "market": "NASDAQ", "period": period, "predictions": open2 }]}}
    #r = requests.post(url, json.dumps(payload), headers=headers)
    #if r.status_code == 200:
    #    print 'success'
    #else:
    #    print 'error: ' + str(r.status_code) + '\n\n' + r.text
    #return open,high,low,close
    results.append(open)
    results.append(high)
    results.append(low)
    results.append(close)
    
    results2 = pd.Series(results).to_json(orient='values')
    payload = {"type": "insert", "args":{"table": "stockTest", "objects": [{"stocksymbol": stock_name, "market": "NASDAQ", "period": period, "predictions": results2 }]}}
    r = requests.post(url, json.dumps(payload), headers=headers)
    if r.status_code == 200:
        print 'success'
    else:
        print 'error: ' + str(r.status_code) + '\n\n' + r.text
    #results2 = str(results)
    #results = np.hstack([open,high,low,close])
    #results = np.append(open, high, low, close)
    #print results



# add model. predict return the output to result
url = 'https://data.chastiser11.hasura-app.io/v1/query'


headers = {"content-type": "application/json", "authorization": "Bearer wj7fmf21w6lvu0l4u7vmdef1tqo0cykn"}

# This will get passed by the front end, on converting from speech to text
symbol="VMWare"

payload = {"type": "select", "args":{"table": "companyDB", "columns": ["stocksymbol"],
        "where": { "name": symbol} }}
        

        
r = requests.post(url, json.dumps(payload), headers=headers)
stock_symbol = json.loads(r.text)

print stock_symbol

if r.status_code == 200:
    print 'success'
    predict_stock(stock_symbol[0]['stocksymbol'],10)
else:
    print 'error: ' + str(r.status_code) + '\n\n' + r.text

#        break

#payload = {"type": "insert", "args":{"table": "stockforecast", "objects": [{"stocksymbol": stock_symbol, "market": "NASDAQ", "period": period, "predictions": results2 }]}}


#r = requests.post(url, json.dumps(payload), headers=headers)

#stock_name = stock_symbol[0]['stocksymbol']

#payload = {"type": "insert", "args":{"table": "stockTest", "objects": [{"stocksymbol": stock_name, "market": "NASDAQ", "period": period, "predictions": results2 }]}}
#r = requests.post(url, json.dumps(payload), headers=headers)

#if r.status_code == 200:
#    print 'success'
#else:
#    print 'error: ' + str(r.status_code) + '\n\n' + r.text
