import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import pandas as pd
from pandas import datetime
import math, time
import itertools
#import requests
from urllib2 import Request, urlopen, URLError
from sklearn import preprocessing
import datetime
#from newsapi.articles import articles


API_KEY="a701f01e365040729efbef745a40f395"

key = '96af62a035db45bda517a9ca62a25ac3'

parameters = {"source": 'the-next-web', "apiKey": {API_KEY}}


request = Request('https://newsapi.org/v1/sources')
response = urlopen(request)
#print response.read()
#print kittens


sources = Request("https://newsapi.org/v1/sources")

print sources

#print response.read()

#response = Request("https://newsapi.org/v1/articles", params=parameters)

#print response

import json, requests

url = 'https://newsapi.org/v1/articles'

params = dict(
              source = 'the-next-web',
              apiKey = {key}
              )



resp = requests.get(url=url, params=params)
data = json.loads(resp.text)

#info = json.loads(data)
#print json.dumps(info,indent=4)
lst = list()
for item in data['articles']:
    sub = item['description']
    lst.append(sub)
    with open('data3.txt', 'a+') as outfile:
        outfile.write(sub.encode('ascii', 'ignore'))
print lst


#sources[0]['id']

#with open('data2.json', 'w') as outfile:
#    outfile.write(resp.text.encode('ascii', 'ignore'))


#print data['articles']['description']

