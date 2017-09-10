import pandas_datareader.data as web
import h5py
import datetime

stocks = ['VMW','SPLK', 'CDNS', 'GOOG', 'FB', 'NVDA', 'AAPL', 'AMZN', 'MSFT']

start = datetime.datetime(2017, 8, 9)
print start
end = datetime.date.today()
print end

for stock in stocks:
    df = web.DataReader(stock, "yahoo", start, end)
    df.to_csv(stock+'.csv', index=True)
