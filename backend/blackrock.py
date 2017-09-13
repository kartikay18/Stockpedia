import requests
import numpy as np
import json
import urllib2
import pandas as pd
from newsapi import *
#import nltk
import sys, csv, json
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *


portfolioAnalysisRequest = requests.get("https://www.blackrock.com/tools/hackathon/portfolio-analysis", params={'positions' : 'BLK~25|AAPL~25|IXN~25|MALOX~25','fromDate': "20101212", 'toDate': "20161212"})


#print (portfolioAnalysisRequest.text) # get in text string format
#print json.dumps(portfolioAnalysisRequest.json) # get as json object
with open('data.json', 'w') as outfile:
   outfile.write(portfolioAnalysisRequest.text.encode('ascii', 'ignore'))
