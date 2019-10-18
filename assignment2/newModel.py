# import some library
# -*- coding: utf-8 -*
import numpy as np
import string
import pandas as pd
from sklearn.datasets import make_classification as mk
# model
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from xgboost import plot_importance, XGBClassifier
from sklearn.preprocessing import OrdinalEncoder

from naiveBayes import Berboulli_Naive_Bayes
from sklearn.ensemble import BaggingClassifier as BC, RandomForestClassifier

# help_clean
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
# help_feature_process
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import bigrams
# help_analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# analysis
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import xgboost as xgb

class Reader:

    def read(self,path):
        data = pd.read_csv(path, encoding="utf-8")
        return data

    def shuffle(self,df):
        df.shuffle()

    def write(self,name):
        f = open(name, "w")
        for line in self.data:
            f.write(line)
        f.close()

    def extractColToString(self,df,col_name):
        data_p = [ (line) for line in self.file]
        return data_p

# 2 Cleaning of the data
class Cleaner:
    def __init__(self, sample_list,use_lemmer,use_stemmer, use_stopwords):
        self.sents_list = sample_list
        self.words_list = [self.splitter(w) for w in sample_list]
        self.s =use_stemmer
        self.l =use_lemmer
        self.st = use_stopwords

    def splitter(self,sample_list):
        pos_words = sample_list.split()
        return pos_words

    def remove_punc(self):
        removed_punc = []
        table = str.maketrans('', '', string.punctuation)
        for s in self.words_list:
            removed_punc.append( [w.translate(table) for w in s] )
        self.words_list = removed_punc

    def lowercase(self):
        lowered = []
        for s in self.words_list:
            lowered.append( [w.lower() for w in s])
        self.words_list = lowered

    def remove_noncharacter(self):
        remove_nonchar = []
        for s in self.words_list:
            remove_nonchar.append([w for w in s if w.isalnum()])
        self.words_list = remove_nonchar

    def remove_stopWord(self):
        removed_stop = []
        stop_words = stopwords.words('english')
        for s in self.words_list:
            removed_stop.append([w for w in s if not w in stop_words])
        self.words_list = removed_stop

    def lemmatizer(self):
        lemmatized = []
        lemmatizer = WordNetLemmatizer()
        for s in self.words_list:
            lemmatized.append([lemmatizer.lemmatize(w) for w in s])
        self.words_list = lemmatized

    def stemmer(self):
        stemmed = []
        porter = PorterStemmer()
        for s in self.words_list:
            stemmed.append( [porter.stem(word) for word in s])
        self.words_list = stemmed

    def clean_low_puc_nc_le_stop(self):
        cleaned = []
        #porter = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words('english')
        table = str.maketrans('', '', string.punctuation)
        for s in self.words_list:
            cleaned.append([lemmatizer.lemmatize(word.translate(table).lower()) for word in s if word not in stop_words])
        self.words_list = cleaned

    def cleaned(self):
        #self.lowercase()
        self.remove_punc()
        #self.remove_noncharacter()
        if self.l:
            self.lemmatizer()
        if self.s:
            self.stemmer()
        if self.st:
            self.remove_stopWord()
        result = self.joined()
        return result

    def joined(self):
        sents = []
        for s in self.words_list:
            sents.append(' '.join(s))
        return sents
# 3 feature processing

class Feature_Processer:

    def split(self,features_set,target_set, ratio):
        X_train, X_test, y_train, y_test = train_test_split(features_set, target_set, train_size=ratio,
                                                            test_size=1-ratio)
        return X_train, X_test, y_train, y_test
    #n_grams, min_df
    #adjustable (1,2) is not good as (1,1)
    def count_vector_features_produce(self, X_train, X_test, thresold):
        cv = CountVectorizer(binary=True,min_df=thresold)
        cv.fit(X_train)
        X = cv.transform(X_train)
        X_test = cv.transform(X_test)
        return X, X_test

    def tf_idf(self,X_train,X_test,n_grams,thresold):
        tf_idf_vectorizer = TfidfVectorizer(ngram_range=n_grams,min_df =thresold,binary=True)
        vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
        vectors_test_idf = tf_idf_vectorizer.transform(X_test)
        return vectors_train_idf,vectors_test_idf


class classifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


class main:
    data_raw = Reader().read("reddit_train.csv")
    data_train = data_raw['comments']
    data_test = data_raw['subreddits']
    # use_lemmer,use_stemmer, use_stopwords
    cleaner_train = Cleaner(data_train, True, False, False)
    cleaner_train.cleaned()

    X_train, X_test, y_train, y_test = Feature_Processer().split(data_train, data_test, 0.9)
    X_train, X_test = Feature_Processer().tf_idf(X_train, X_test, (1, 1), 1)

    #clf = classifier(X_train, X_test, y_train, y_test)
    #enc = OrdinalEncoder()
    #X = [['worldnews', 1], ['canada', 2], ['AskReddit', 3],['wow',4],['conspiracy',5],
    #     ['nba',6],['leagueoflegends',7],['soccer',8],['funny',9],['movies',10],
    #     ['anime',11],['Overwatch',12],['trees',13],['GlobalOffensive',14],['nfl',15],
    #    ['europe',16],['Music',20],['baseball',17],['hockey',18],['gameofthrones',19]]
    #enc.fit(X)
   # y_train_matrix= y_train.values.reshape(len(y_train),1)

    #y_test_matrix = y_test.values.reshape(len(y_test), 1)
    #y_train_matrix = enc.transform(y_train_matrix)
    #y_test_matrix=enc.transform(y_test_matrix)
    #print(y_train_matrix)
    #clf.multNB()
    # read in data

    #model = XGBClassifier()

    #dtrain = xgb.DMatrix(X_train,y_train_matrix)
    #dtest = xgb.DMatrix(X_test, y_test_matrix)
    # specify parameters via map
    param = {'booster': 'gbtree','min_child_weight':2,'max_depth':6,'num_class': 20,'lambda':2, 'eta': 0.3, 'silent': 0, 'objective': 'multi:softmax'}
    num_round = 2
    #kf = KFold(n_splits=5, shuffle=True, random_state=1)
    #for train_index, test_index in kf.split(X_train):

    xgb_model = xgb.XGBClassifier(booster='gbtree',min_child_weight=2,objective='multi:softmax',n_estimators=150,eta= 0.3,n_jobs=-1).fit(X_train, y_train)
    predictions = xgb_model.predict(X_test)
    actuals = y_test
    #print(confusion_matrix(actuals, predictions))

    print("bst  : accurancy_is", metrics.accuracy_score(actuals,predictions))

if __name__ == "__main__":
    main()