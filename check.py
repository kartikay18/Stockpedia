
import pandas_datareader.data as web
import h5py
import datetime

stocks = ['VMW','SPLK', 'CDNS', 'GOOG', 'FB', 'NVDA', 'AAPL', 'AMZN', 'MSFT']

start = datetime.datetime(2015, 1, 1)
end = datetime.date.today()

for stock in stocks:
    df = web.DataReader(stock, "yahoo", start, end)
    if df is not None:
        print stock
