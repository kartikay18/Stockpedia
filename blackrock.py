import requests
import numpy as np
import pandas as pd
#import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *


"""
    For the examples we are using 'requests' which is a popular minimalistic python library for making HTTP requests.
    Please use 'pip install requests' to add it to your python libraries.
    """

portfolioAnalysisRequest = requests.get("https://www.blackrock.com/tools/hackathon/portfolio-analysis", params={'positions' : 'BLK~25|AAPL~25|IXN~25|MALOX~25','fromDate': "20101212", 'toDate': "20161212"})
print (portfolioAnalysisRequest.text) # get in text string format
print (portfolioAnalysisRequest.json) # get as json object
