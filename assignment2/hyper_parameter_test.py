# 模型创建函数，KerasClassifier需要这个函数
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from xgboost import plot_importance, XGBClassifier
from sklearn.linear_model import SGDClassifier as SGD


class Reader:

    def read(self, path):
        data = pd.read_csv(path, encoding="utf-8")
        return data

    def shuffle(self, df):
        df.shuffle()

    def extractColToString(self, df, col_name):
        data_p = [(line) for line in self.file]
        return data_p

class Feature_Processer:
    def split(self, features_set, target_set, ratio):
        X_train, X_test, y_train, y_test = train_test_split(features_set, target_set, train_size=ratio,
                                                            test_size=1 - ratio, random_state=14)
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

    def SGD(self, alpha, penalty):
        model = SGD( alpha = alpha, penalty = penalty)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)

        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of LDA in Cross Validation", scores3.mean() * 100)

        # print(" SGD : accurancy_is", metrics.accuracy_score(self.y_test, pred))
        return pred

    def random_forest(self):
        clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2,n_jobs=-1)
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_test)
        return y_pred
if __name__ == '__main__':

    data_raw = Reader().read("reddit_train.csv")
    data_train = data_raw['comments']
    data_test = data_raw['subreddits']
    # use_lemmer,use_stemmer, use_stopwords
    # cleaner_train = Cleaner(data_train, True, False, False)
    # data_train = cleaner_train.cleaned()

    #X_train, X_test, y_train, y_test = Feature_Processer().split(data_train, data_test, 1)

    tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
    tf_idf_vectorizer.fit(data_train)

    X_train_tf = tf_idf_vectorizer.transform(data_train)
    #print(X_train_tf[0:1])

    #print(X_test_tf[0:1])
    scores = ['precision_macro']
    parameter_space = {
        'min_child_weight':[2],
        'n_estimators':[50],
        'eta': [0.1]
     }

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = XGBClassifier(booster='gbtree',objective='multi:softmax')
        grid = GridSearchCV(clf, parameter_space, cv=5, scoring='%s' % score)
        grid.fit(X_train_tf, data_test)

        print("Best parameters set found on development set:")
        print()
        print(grid.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = grid.cv_results_['mean_test_score']
        stds = grid.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))