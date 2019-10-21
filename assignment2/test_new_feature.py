import string
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn import metrics
from scipy.sparse import hstack


class Reader:

    def read(self, path):
        data = pd.read_csv(path, encoding="utf-8")
        return data

class Cleaner:
    def __init__(self, sample_list):
        self.sents_list = sample_list
        self.words_list = [self.splitter(w) for w in sample_list]

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

    def joined(self):
        sents = []
        for s in self.words_list:
            sents.append(' '.join(s))
        return sents


class Feature_Processer:
    def split(self, features_set, target_set, ratio):
        X_train, X_test, y_train, y_test = train_test_split(features_set, target_set, train_size=ratio,
                                                            test_size=1 - ratio,shuffle=False)
        return X_train, X_test, y_train, y_test


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


    def multNB(self, alpha):
        model = MultinomialNB(alpha=alpha)
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of multiple", scores3.mean() * 100)
        print("Score of multiple", metrics.accuracy_score(self.y_test, preds))


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

if __name__ == '__main__':
    data_raw = Reader().read("reddit_train.csv")
    data_train = data_raw['comments']
    data_test = data_raw['subreddits']

    c = Cleaner(data_train)
    c.lowercase()
    c.remove_punc()
    data_train = c.joined()

    # print(data_train[0:1])
    seperate_class_list, Z_score= drawer().drawing(data_train, data_test)

    X_train, X_test, y_train, y_test = Feature_Processer().split(data_train, data_test, 0.9)
    tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
    tf_idf_vectorizer.fit(X_train)
    X_train_tf = tf_idf_vectorizer.transform(X_train)
    X_test_tf = tf_idf_vectorizer.transform(X_test)

    clf = classifier(X_train_tf, X_test_tf, y_train, y_test)
    print("Experiment1")
    clf.multNB(alpha=0.2)

    X_train_dtm = hstack((X_train_tf,np.array(Z_score[:len(X_train)])[:,None]))
    X_test_dtm = hstack((X_test_tf, np.array(Z_score[:len(X_test)])[:, None]))


    clf2 = classifier( X_train_dtm, X_test_dtm, y_train, y_test)
    print("Experiment2")
    clf2.multNB(alpha=0.2)
