# import some library
# -*- coding: utf-8 -*
import numpy as np
import string
import pandas as pd
#model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
#help_clean
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#help_feature_process
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import bigrams
#help_analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#analysis
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from statistics import mean

#read from the file, possible write for testing
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
        self.lowercase()
        self.remove_punc()
        self.remove_noncharacter()
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

def main():
    data_raw = Reader().read("reddit_train.csv")
    data_train = data_raw['comments']
    data_test = data_raw['subreddits']
    #use_lemmer,use_stemmer, use_stopwords
    cleaner_train = Cleaner(data_train,True,False,True)
    cleaner_train.cleaned()

    X_train, X_test, y_train, y_test = Feature_Processer().split(data_train,data_test,0.8)
    X_train, X_test = Feature_Processer().tf_idf(X_train, X_test,(1,2),1)

    clf = classifier(X_train, X_test, y_train, y_test)
    #logistic converges deadly
    #clf.logistic(1.0)
    clf.svm(0.2)
    #跑不动
    #clf.decision_tree()
    #clf.QDA()
    clf.multNB()
    #svm approximately 56%
    #multinomial Nb with 54%


if __name__ == "__main__":
    main()
