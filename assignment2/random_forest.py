# import some library
# -*- coding: utf-8 -*
import numpy as np
import string
import pandas as pd
from sklearn.datasets import make_classification as mk
# model
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from naiveBayes import Berboulli_Naive_Bayes
from sklearn.ensemble import BaggingClassifier as BC, RandomForestClassifier

# help_clean
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
# help_feature_process
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
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
        tf_idf_vectorizer = TfidfVectorizer(ngram_range=n_grams,min_df =thresold)
        vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
        vectors_test_idf = tf_idf_vectorizer.transform(X_test)
        return vectors_train_idf,vectors_test_idf


class classifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def random_forest(self):

        #parameter_space = {
        #    "n_estimators": [150],
        #    "criterion": ["gini"],
        #    "min_samples_leaf": [1,2],
        #}

        # scores = ['precision', 'recall', 'roc_auc']
        #scores = ['precision_macro']
        clf = RandomForestClassifier(random_state=14,n_estimators=150,min_samples_leaf=2)
        #for score in scores:
        #    print("# Tuning hyper-parameters for %s" % score)
        #    print()

        #    clf = RandomForestClassifier(random_state=14)
        #    grid = GridSearchCV(clf, parameter_space, cv=5, scoring='%s' % score)
            # scoring='%s_macro' % score：precision_macro、recall_macro是用于multiclass/multilabel任务的
        #    grid.fit(self.x_train, self.y_train)

        #    print("Best parameters set found on development set:")
        #    print()
        #    print(grid.best_params_)
        #    print()
        #    print("Grid scores on development set:")
        #    print()
        #    means = grid.cv_results_['mean_test_score']
        #    stds = grid.cv_results_['std_test_score']
        #    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        #        print("%0.3f (+/-%0.03f) for %r"
        #              % (mean, std * 2, params))

        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        return y_pred
        #scores3 = cross_val_score(clf, self.x_train, self.y_train, cv=5, scoring='accuracy')
        #print("Score of decision tree in Cross Validation", scores3.mean() * 100)
        #y_pred = clf.predict(self.x_test)
        #print("random forest  : accurancy_is", metrics.accuracy_score(self.y_test,y_pred))

class main:
    data_raw = Reader().read("reddit_train.csv")
    data_train = data_raw['comments']
    data_test = data_raw['subreddits']

    cleaner_train = Cleaner(data_train, False, False, False)
    cleaner_train.cleaned()

    X_train, X_test, y_train, y_test = Feature_Processer().split(data_train, data_test, 0.9)
    X_train, X_test = Feature_Processer().tf_idf(X_train, X_test, (1, 1), 1)

    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    clf = classifier(X_train, X_test, y_train, y_test)
    clf.random_forest()

if __name__ == "__main__":
    main()
