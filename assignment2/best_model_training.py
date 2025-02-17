# import some library
# -*- coding: utf-8 -*
import numpy as np
import string
import pandas as pd
from numpy import hstack
from sklearn.datasets import make_classification as mk
# model
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier as BC, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_classification as mc
from sklearn.linear_model import SGDClassifier as SGD
# help_clean
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
# help_feature_process
from sklearn.model_selection import train_test_split
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
from statistics import mean
from collections import Counter


# read from the file, possible write for testing
class Reader:

    def read(self, path):
        data = pd.read_csv(path, encoding="utf-8")
        return data

    def shuffle(self, df):
        df.shuffle()

    def extractColToString(self, df, col_name):
        data_p = [(line) for line in self.file]
        return data_p


# 2 Cleaning of the data
class Cleaner:
    def __init__(self, sample_list, use_lemmer, use_stemmer, use_stopwords):
        self.sents_list = sample_list
        self.words_list = [self.splitter(w) for w in sample_list]
        self.s = use_stemmer
        self.l = use_lemmer
        self.st = use_stopwords

    def splitter(self, sample_list):
        pos_words = sample_list.split()
        return pos_words

    def remove_punc(self):
        removed_punc = []
        table = str.maketrans('', '', string.punctuation)
        for s in self.words_list:
            removed_punc.append([w.translate(table) for w in s])
        self.words_list = removed_punc

    def lowercase(self):
        lowered = []
        for s in self.words_list:
            lowered.append([w.lower() for w in s])
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
            stemmed.append([porter.stem(word) for word in s])
        self.words_list = stemmed


    def cleaned(self):
        #self.lowercase()
        # self.remove_punc()
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
    def split(self, features_set, target_set, ratio):
        X_train, X_test, y_train, y_test = train_test_split(features_set, target_set, train_size=ratio,
                                                            test_size=1 - ratio)
        return X_train, X_test, y_train, y_test

    # n_grams, min_df
    # adjustable (1,2) is not good as (1,1)
    def count_vector_features_produce(self, X_train, X_test, thresold):
        cv = CountVectorizer(binary=True, min_df=thresold)
        cv.fit(X_train)
        X = cv.transform(X_train)
        X_test = cv.transform(X_test)
        return X, X_test

    def tf_idf(self, X_train, X_test, n_grams, thresold):
        tf_idf_vectorizer = TfidfVectorizer(ngram_range=n_grams, min_df=thresold)
        vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
        print()
        vectors_test_idf = tf_idf_vectorizer.transform(X_test)
        return vectors_train_idf, vectors_test_idf


class classifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def logistic(self, c, epochs):
        model = LogisticRegression(C=c, dual=False, solver='saga', multi_class='multinomial', max_iter=epochs)
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)

        return preds

    def svm(self, c):
        # n_estimators = 10
        model = LinearSVC(C=c, class_weight='balanced')
        print("start fitting")
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of svm", scores3.mean() * 100)
        print("Score of svm", metrics.accuracy_score(self.y_test, preds))
        return preds

    def multNB(self, alpha):
        model = MultinomialNB(alpha=alpha)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        return pred

    def KNeighbors(self, iter):
        model = KNeighborsClassifier(n_neighbors=iter, weights='uniform', algorithm='auto', n_jobs=-1)
        model.fit(self.x_train, self.y_train)
        predict = model.predict(self.x_test)
        # print(" kneighbors Regression : accurancy_is", metrics.accuracy_score(self.y_test, predict))
        return predict

    def Ber_NaiveBayes(self, alpha):
        model = BernoulliNB(alpha=alpha).fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        return preds

    def SGD(self, alpha, penalty):
        model = SGD( alpha = alpha, penalty = penalty)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)

        # scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        # print("Score of LDA in Cross Validation", scores3.mean() * 100)

        # print(" SGD : accurancy_is", metrics.accuracy_score(self.y_test, pred))
        return pred

    def random_forest(self):
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,n_jobs=-1)
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        return y_pred
class drawer:
    def drawing(self, sents_list, tag_list):
        # 20 class, 20 graph
        unique_tag = ['worldnews', 'canada', 'AskReddit', 'wow', 'conspiracy',
                      'nba', 'leagueoflegends', 'soccer', 'funny', 'movies',
                      'anime', 'Overwatch', 'trees', 'GlobalOffensive', 'nfl',
                      'europe', 'Music', 'baseball', 'hockey', 'gameofthrones']
        seperate_class_list = []
        record_index = []
        for tag in unique_tag:
            same_class_list = []
            for i in range(len(tag_list)):
                if (tag_list[i] == tag):
                    # cut by the word
                    same_class_list.append(len(sents_list[i].split(" ")))
                    record_index.append(i)
            seperate_class_list.append(same_class_list)

        df = pd.DataFrame.from_records(seperate_class_list, unique_tag)
        df = df.T
        statics = df.describe()
        Z_score = []
        for j in range(len(tag_list)):
            mean = statics.loc["min", tag_list[j] ]
            std = statics.at["std", tag_list[j] ]
            Z_score.append((len(sents_list[j].split(" "))-mean)/std)

        return seperate_class_list,Z_score


def main():
    data_raw = Reader().read("reddit_train.csv")

    data_train = data_raw['comments']
    data_test = data_raw['subreddits']
    # use_lemmer,use_stemmer, use_stopwords
    cleaner_train = Cleaner(data_train, True, False, False)
    cleaner_train.cleaned()

    X_train, X_test, y_train, y_test = Feature_Processer().split(data_train, data_test, 0.9, True)
    X_train, X_test = Feature_Processer().tf_idf(X_train, X_test, (1, 1), 1)

    clf = classifier(X_train, X_test, y_train, y_test)
    for k in [100, 150, 200, 250, 300]:
        print('k is', k)
        clf.KNeighbors(k)


def stack():
    # ----------    USED FOR STACKED ENSEMBLE USING VOTING ------
    data_raw = Reader().read("reddit_train.csv")
    data_train = data_raw['comments']
    data_test = data_raw['subreddits']
    # use_lemmer,use_stemmer, use_stopwords
    cleaner_train = Cleaner(data_train, False, True, True)
    data_train = cleaner_train.cleaned()

    X_train, X_test, y_train, y_test = Feature_Processer().split(data_train,data_test,0.9)


    tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
    tf_idf_vectorizer.fit(X_train)

    X_train_tf = tf_idf_vectorizer.transform(X_train)
    X_test_tf = tf_idf_vectorizer.transform(X_test)

    seperate_class_list, Z_score= drawer().drawing(data_train, data_test)
    X_train_dtm = hstack((X_train_tf,np.array(Z_score[:len(X_train)])[:,None]))
    X_test_dtm = hstack((X_test_tf, np.array(Z_score[:len(X_test)])[:, None]))

    clf = classifier(X_train_tf, X_test_tf, y_train, y_test)
    #clf = classifier(X_train_dtm,X_test_dtm,y_train,y_test)
    svm_pred = clf.svm(0.2)
    multNB1_pred = clf.multNB(alpha=0.2)
    log_pred = clf.logistic(3, 1200)
    SDG = clf.SGD(alpha=5e-05, penalty='l2')
    kn_pred = clf.KNeighbors(110)
    rf_pred = clf.random_forest()

    X_train_bi, X_test_bi = Feature_Processer().count_vector_features_produce(X_train,X_test,1)
    clf2 = classifier(X_train_bi, X_test_bi, y_train, y_test)
    numNB2_pred = clf2.multNB(alpha=0.3)
    bnb = clf2.Ber_NaiveBayes(alpha=0.024)

    voted = []

    for i in range(len(svm_pred)):
        group = [svm_pred[i],svm_pred[i], multNB1_pred[i], multNB1_pred[i],log_pred[i], numNB2_pred[i],numNB2_pred[i],bnb[i],
                 SDG[i],kn_pred[i],rf_pred[i]]
        c = Counter(group)
        value, count = c.most_common()[0]
        voted.append(value)

    print("Stacked : accurancy_is", metrics.accuracy_score(y_test, voted))

if __name__ == "__main__":
    # main()
    stack()

