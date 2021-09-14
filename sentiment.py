# import the libraries
#!pip install keras
#!pip install tensorflow
from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import pandas as pd

# Get The Data
data = pd.read_csv("train.tsv", sep="\t")
#data.head()

# Edit The Data
data["Sentiment"].replace(0, value="negatif", inplace=True)
data["Sentiment"].replace(1, value="negatif", inplace=True)
data["Sentiment"].replace(3, value="pozitif", inplace=True)
data["Sentiment"].replace(4, value="pozitif", inplace=True)
#data.head()
data = data[(data.Sentiment == "negatif") | (data.Sentiment == "pozitif")]
#data.head()

# Group The Data
data.groupby("Sentiment").count()
