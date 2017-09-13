import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import pandas as pd
import json
import httplib, urllib, urllib2
from pandas import datetime
import math, time
import itertools
from urllib2 import Request, urlopen, URLError
import datetime
import json, requests
#from newsapi.articles import articles


API_KEY="a701f01e365040729efbef745a40f395"
key = '96af62a035db45bda517a9ca62a25ac3'
parameters = {"source": 'the-next-web', "apiKey": {API_KEY}}



request = Request('https://newsapi.org/v1/sources')
response = urlopen(request)
#print response.read()

sources = Request("https://newsapi.org/v1/sources")
# print sources

url = 'https://newsapi.org/v1/articles'

publisher = 'the-next-web'
params = dict(
              source = publisher,
              sortBy = 'latest',
              apiKey = {key}
              )

resp = requests.get(url=url, params=params)
data = json.loads(resp.text)

#info = json.loads(data)
#print json.dumps(info,indent=4)
lst = list()
for item in data['articles']:
    sub = item['description']
    lst.append((sub))
    with open('data3.txt', 'a+') as outfile:
        outfile.write(sub.encode('ascii', 'ignore'))
print type(lst)



# HTTP request to a sent analysis API

params2 = urllib.urlencode({
                           'text' : {resp.text}
                          })


url2 = 'http://text-processing.com/api/sentiment/'
request = urllib2.Request(url2, params2)

response = urllib2.urlopen(request)
# print response.read()
respSent = json.loads(response.read())

url3 = 'https://data.chastiser11.hasura-app.io/v1/query'
headers = {"content-type": "application/json", "authorization": "Bearer wj7fmf21w6lvu0l4u7vmdef1tqo0cykn"}
# resp = json.loads(r.text)
payload = {"type": "insert", "args":{"table": "sentiment", "objects": [{"newsoutlet": publisher, "neg": respSent['probability']['neg'], "pos": respSent['probability']['pos'], "label": respSent['label'], "articles": json.dumps(lst) }] }}
r = requests.post(url3, json.dumps(payload), headers=headers)
# print r.status_code
# print r.text

# print response.read()



#encoding = response.info().get_content_charset('utf8')
#data = json.loads(response.read().decode(encoding))

#urllib2.urlopen("http://text-processing.com/api/sentiment/").read()

#sources[0]['id']

#with open('data2.json', 'w') as outfile:
#    outfile.write(resp.text.encode('ascii', 'ignore'))


#print data['articles']['description']