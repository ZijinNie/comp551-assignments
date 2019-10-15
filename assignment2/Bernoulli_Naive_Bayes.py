# import some library
# -*- coding: utf-8 -*
import numpy as np
import string
import pandas as pd
from sklearn.datasets import make_classification as mk
#model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from naiveBayes import Berboulli_Naive_Bayes
from sklearn.ensemble import BaggingClassifier as BC

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
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix


#read from the file, possible write for testing
class Reader:

    def read(self,path):
        data = pd.read_csv(path, encoding="utf-8")
        return data

    def shuffle(self,df):
        df.shuffle()

    #TODO: write to file latter
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
    def __init__(self, x_train, x_test, y_train,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def logistic(self, c,epochs):
        #,max_iter = epochs
        model = LogisticRegression(C=c, dual=False, solver='lbfgs',multi_class= 'multinomial')
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        scores1 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of Logistic in Cross Validation", scores1.mean() * 100)
        print("Losistic Regression : accurancy_matrix is", metrics.accuracy_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        #print("Confusion Matrix\n", cm)
        #print("Report", classification_report(self.y_test, preds))
        
    def SelfNaiveByes(self):
        model = Berboulli_Naive_Bayes()
        model.train(self.x_train,self.y_train)
        print(model.score(self.x_test,self.y_test))
        
    def Ber_NaiveBayes(self, alpha):
        model = BernoulliNB(alpha=alpha).fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        scores2 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of Naive Bayes", scores2.mean() * 100)
        print("Bernoulli Naive Bayes : accurancy_matrix is", metrics.accuracy_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        #print("Confusion Matrix\n", cm)
        #print("Report", classification_report(self.y_test, preds))

    def svm(self, c):
        model = LinearSVC(C=c)
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)

        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of SVM in Cross Validation", scores3.mean() * 100)
        print("SVM Regression : accurancy_is", metrics.accuracy_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        #print("Confusion Matrix\n", cm)
        #print("Report", classification_report(self.y_test, preds))

    def decision_tree(self):
        #criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False
        model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,min_samples_split=0.1)
        preds =model.fit(self.x_train, self.y_train)
        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of decision tree in Cross Validation", scores3.mean() * 100)
        print("decision tree  : accurancy_is", metrics.accuracy_score(self.y_test, model.predict(self.x_test)))
        cm = confusion_matrix(self.y_test, preds)
        # print("Confusion Matrix\n", cm)
        # print("Report", classification_report(self.y_test, preds))

    def dummy(self):
        clf = DummyClassifier(strategy='stratified', random_state=0)
        clf.fit(self.x_train, self.y_train)
        score = clf.score(self.x_test, self.y_test)
        print("Random Baseline's accurancy", score)

    def multNB(self):
        model = MultinomialNB()
        preds = model.fit(self.x_train, self.y_train)
        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5, scoring='accuracy')
        print("Score of MultinomialNB in Cross Validation", scores3.mean() * 100)
        print(" MultinomialNB Regression : accurancy_is", metrics.accuracy_score(self.y_test, model.predict(self.x_test)))
        cm = confusion_matrix(self.y_test, preds)
        # print("Confusion Matrix\n", cm)
        # print("Report", classification_report(self.y_test, preds))
    


def main():
    data_raw = Reader().read("reddit_train.csv")
    data_train = data_raw['comments']
    data_test = data_raw['subreddits']
    #use_lemmer,use_stemmer, use_stopwords
    cleaner_train = Cleaner(data_train,True,False,False)
    cleaner_train.cleaned()
    
    X_train, X_test, y_train, y_test = Feature_Processer().split(data_train,data_test,0.9)
    X_train, X_test = Feature_Processer().tf_idf(X_train, X_test,(1,1),1)
    
    #This is for running Multi Bernoulli NB
    #categoryDict = dict([(y,x+1) for x,y in enumerate(sorted(set(y_train)))])
    #categoryDict = {'AskReddit': 1, 'GlobalOffensive': 2, 'Music': 3, 'Overwatch': 4, 'anime': 5, 'baseball': 6, 'canada': 7, 'conspiracy': 8, 'europe': 9, 'funny': 10, 'gameofthrones': 11, 'hockey': 12, 'leagueoflegends': 13, 'movies': 14, 'nba': 15, 'nfl': 16, 'soccer': 17, 'trees': 18, 'worldnews': 19, 'wow': 20}
    #y_train = [categoryDict[i] for i in y_train]
    #y_test = [categoryDict[i] for i in y_test]
    #X_train, X_test = Feature_Processer().count_vector_features_produce(X_train,X_test,1)
    #X_train = X_train.toArray()
    #X_test = X_test.toArray()
    #clf.SelfNaiveByes()

    clf = classifier(X_train, X_test, y_train, y_test)
    a = [10,100,1000]
    for x in a:
        print("Current a:",x)
        clf.logistic(x,1000)
    #a = 10 53% acc 55%, 100 52% 55% ,1000 51% acc 54%


    #a = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    #for x in a:
        #print("Current a:",x)
    clf.svm(x)
    #svm approximately 56-57% using tf_idf, 54-55% using binary_vetorizor
    #svm 0.001 42% 0.01 50-51% 0.1 56% 1 55% 10 49
    #svm 0.1 55.8, 0.2 56.25,0.3 56.17,0.4 55.98,0.5 55.76

    #clf.decision_tree()
    #Decision tree 23% 0.01 28%
    #Decision tree     0.001 27%
    #Decision tree     0.1 26%

    #clf.multNB()
    #multinomial Nb with 55-56% for removing the frequency less than 2 20% 0.0001 54%
    #bigram with unigram 50% distrucbution 2 51%-52%  0.02 20%

if __name__ == "__main__":
    main()
