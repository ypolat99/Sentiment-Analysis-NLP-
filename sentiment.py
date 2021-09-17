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

#df.head()
#df.iloc[0]


# Seperate into TRAIN and TEST Parts

train_x, test_x, train_y, test_y = model_selection.train_test_split(df["text"],df["label"], random_state=1)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)
# train_y[0:5]


# ADD VECTORIZERS

#Count Vectors
vectorizer = CountVectorizer()
vectorizer.fit(train_x)
x_train_count = vectorizer.transform(train_x)
x_test_count = vectorizer.transform(test_x)
vectorizer.get_feature_names()[0:5]

#tf idfg --> word level
tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

#n gram level tf-idf
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range = (2,3))
tf_idf_ngram_vectorizer.fit(train_x)
x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)

# character level tf -idf
tf_idf_chars_vectorizer = TfidfVectorizer(analyzer = "char", ngram_range = (2,3))
tf_idf_chars_vectorizer.fit(train_x)
x_train_tf_idf_chars = tf_idf_chars_vectorizer.transform(train_x)
x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)




#MACHINE LEARNING PART

# LOGISTIC REGRESSION

#Count Vectors
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_count, test_y, cv=10).mean()
print("Count Vectors Doğruluk Oranı: ", accuracy)

# Word Level TF-IDF
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_word, train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_tf_idf_word, test_y, cv=10).mean()
print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)

#N-GRAM
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_tf_idf_ngram, test_y, cv = 10).mean()
print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)

# Char Level
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(loj_model, x_test_tf_idf_chars, test_y, cv = 10).mean()
print("CHARLEVEL Doğruluk Oranı:", accuracy)

#----------------------------------------------

# NAIVE BAYES
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_count, test_y, cv=10).mean()
print("Count Vectors Doğruluk Oranı:", accuracy)

# Word Level TF-IDF
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(nb_model, x_test_tf_idf_word, test_y, cv = 10).mean()
print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)

#----------------------------------------------

# RANDOM FOREST

# Count Vectores
rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_count, y_train)
accuracy = model_selection.cross_val_score(rf_model, x_test_count, test_y, cv=10).mean()
print("Count Vectors Doğruluk Oranı:", accuracy)

#----------------------------------------------

# XG BOOST

# Count Vectores
xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(xgb_model, x_test_count, test_y, cv = 10).mean()
print("Count Vectors Doğruluk Oranı:", accuracy)



## TESTING WITH THE CHOSEN MODEL

#loj_model.predict("yes i like this film")
new_comment = pd.Series("tihs film is very nice and good i like it")
v = CountVectorizer()
v.fit(train_x)   ## Önemli NOKTA BURASI  !! IMPORTANT PART HERE
new_comment = v.transform(new_comment )
loj_model.predict(new_comment )  ## ---> OUTPUTS array([1]) where 1 == POSITIVE
