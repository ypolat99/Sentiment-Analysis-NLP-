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

# Simplify the Text Portion
df = pd.DataFrame()
df["text"] = data["Phrase"]
df["label"] = data["Sentiment"]


#buyuk-kucuk donusumu
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#noktalama işaretleri
df['text'] = df['text'].str.replace('[^\w\s]','')
#sayılar
df['text'] = df['text'].str.replace('\d','')
#stopwords
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
#seyreklerin silinmesi
sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))
#lemmi
from textblob import Word
#nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 
