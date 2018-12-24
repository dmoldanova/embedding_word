import os
import time
import pickle
import string
import numpy as np

from operator import itemgetter

from nltk.corpus import stopwords as sw

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report


from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

stop_words = {'isn\'t', 'wonn\'t', 'aren\'t', 'n\'t', 'don\'t', 'doesn\'t', 'hasn\'t', 'haven\'t', 
             '\'s', '\'re', '\'m', '\'ll', '\'ve', '\'d'}
stop_words2 = {'a', 'at', 'the'}

# flag - леммитизация/стеминг/ничего
# flag2 - удалять стоп слова/заменять(МОЙ)
class Preprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, flag, flag1, stem, stopwords=None, punct=None, lower=True, strip=True):
        self.flag = flag
        self.flag1 = flag1
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer    = stem
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords or set(sw.words('english'))
        self.punct      = set(punct) if punct else set(string.punctuation)
        
    def fit(self, X, y=None):
        #print('fit')
        return self

    def inverse_transform(self, X):
        #print('inverse_transform')
        return X

    def transform(self, X):
        #print('transform: ', len(X))
        return [
            self.tokenize(sent) for sent in X
        ]
    
    def tokenize(self, sentenses):
        '''
        sentenses = sentenses.lower()
        sentenses = sentenses.strip()
        for stop in stop_words:
            if sentenses.find(stop)!=-1: 
                self.replace_stop_words(sentenses)
                break
        '''
        res = ''
        for token, tag in pos_tag(wordpunct_tokenize(sentenses)):
            token = token.lower() if self.lower else token
            token = token.strip() if self.strip else token
            token = token.strip('_') if self.strip else token
            token = token.strip('*') if self.strip else token

            # If punctuation or stopword, ignore token and continue
            if self.flag1 == 0:
                if token in self.stopwords or all(char in self.punct for char in token):
                    continue
            else:
                if all(char in self.punct for char in token):
                    continue
                if token in self.stopwords:
                    token = self.replace_stop_words(token)

            # Lemmatize or stemming the token and yield
            if self.flag == 0:
                lemma = token
            elif self.flag == 1:
                lemma = self.lemmatize(token, tag)
            elif self.flag == 2:
                lemma = self.stemmer.stem(token)
            res += lemma + ' '
        res = res.strip()
        return res
        

    def lemmatize(self, token, tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform much more accurate WordNet lemmatization.
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)
    
    def stemmer(self, token):
        
        return self.stem.stem(token)
  
    def replace_stop_words(self, token):
        for stop in stop_words:
            if token.find(stop) != -1:
                if stop.find('n\'t') != -1:
                    ind2 = stop.find('n')
                    token = stop[0:ind2] + ' not'
                    break
                else:
                    token = ' be'
                    break
        return token
    
class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self

# BODY
	# Training Data
f = open('train.iob')
f_w = open('data.iob','w')
train_part = str(f.read())
train_part = train_part[4:len(train_part)]
    # split: sentenses in array
train_array = train_part.split('\nBOS ')
    # split: sentenses and lable
train_array = [train_array[i].split('EOS\t') for i in range(len(train_array))]
for i in range(len(train_array)):
    string_array = train_array[i][1].split(' ')
    train_array[i][1] = string_array[len(string_array)-1]
    #print(train_array[i])
train_array = np.array(train_array)
print(train_array.shape)
data = ['' for i in range(train_array.shape[0])]
target = ['' for i in range(train_array.shape[0])]
for i in range(train_array.shape[0]):
    data[i] = train_array[i][0]
    target[i] = train_array[i][1]
    string_ = data[i] + "\n"
    f_w.write(string_)
data = np.array(data)
target = np.array(target)
print(data.shape)
print(target.shape)
print(train_array)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 10)
print(y_train)

pipe = Pipeline([
            ('preprocessor', Preprocessor(0, 0, PorterStemmer())),
            ('vectorizer', CountVectorizer()),
            #('tfidf', TfidfTransformer()),
            #('to_dense', DenseTransformer()), 
            ('to_dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)),
            ('classifier', classifier)])
search_space = [#{'classifier': [MLPClassifier()]},
                {'classifier': [GradientBoostingClassifier()]},
                {'classifier': [GaussianNB()]}
               ]

clf = GridSearchCV(pipe, search_space, cv=5, verbose=0)
best_model = clf.fit(x_train, y_train)
best_model.best_estimator_.get_params()['classifier']
