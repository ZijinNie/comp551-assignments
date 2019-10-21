# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import make_classification as mc

class Berboulli_Naive_Bayes:
    
    def train (self,x_train,y_train):
        X = x_train
        y = y_train
        #get the number of classes
        classNum = np.amax(y)
        print('classNum',classNum)
        
        #for each class, train and find the log_odds
        theta_k = np.ones(classNum)
        theta_j_k = np.ones((classNum,X.shape[1]))
        for k in range(classNum):
            print("run for class",k)
           
            yk = np.array(y)
            #number of class 1 and 0
            y_sum = 0
            
            #categorize the classes into binary
            for i in range(len(yk)):
                if yk[i] == (k+1):
                    y_sum +=1
                    
            
            theta_k[k] = y_sum/ len(yk)
            
            #number of examples where xj =1 and y =1 
            num_j_k = 0
            for j in range(X.shape[1]):
                num_j_k = 0
                for l in range(X.shape[0]):
                    if (yk[l] == k+1) & (X[l,j] == 1):
                        num_j_k +=1
                        
                theta_j_k[k,j] = (num_j_k + 1) / (y_sum + 2)
                print('theta, j, k',j ,k, 'is',theta_j_k[k,j])
            
            
        self.theta_k = theta_k
        self.theta_j_k = theta_j_k
            
        
    def fit (self, x):
        theta_k = self.theta_k                  
        theta_j_k = self.theta_j_k
        
        #Choose the class with highest prob
        result = np.zeros(x.shape[0])
        for i in range(len(x)):
            bestClass = 0
            bestProb = -100000
            
            for k in range(len(theta_k)):
                prob = np.log(theta_k[k])
                for j in range(theta_j_k.shape[1]):
                    prob += x[i][j] *np.log(theta_j_k[k][j]) +(1-x[i][j])*np.log(1-theta_j_k[k][j])
                #print(prob)
                if prob > bestProb:
                    bestClass = k+1
                    bestProb = prob
                    #print("best is ",bestProb)
            result[i] = bestClass
            
        return result
    
    
    def score(self, X_test, Y_test):
        Y_predict = self.fit(X_test)
        #print(Y_predict)
        #print(compareResult)
        count = 0
        print(Y_predict)
        print(Y_test)
        for i in range(len(Y_predict)):
            if Y_predict[i] == Y_test[i]:
                count +=1
        return "score for out NB is ", count/ len(X_test)

class Feature_Processer:
    def split(self,features_set,target_set, ratio,isShuffle):
        X_train, X_test, y_train, y_test = train_test_split(features_set, target_set, train_size=ratio,
                                                            test_size=1-ratio, shuffle = isShuffle)
        return X_train, X_test, y_train, y_test


    def tf_idf(self,X_train,X_test,n_grams,thresold):
        tf_idf_vectorizer = TfidfVectorizer(ngram_range=n_grams,min_df =thresold)
        vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
        print()
        vectors_test_idf = tf_idf_vectorizer.transform(X_test)
        return vectors_train_idf,vectors_test_idf

class classifier:
    def __init__(self, x_train, x_test, y_train,y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def multNB(self, alpha):
        # n_estimators = 10
        # model = BC( MultinomialNB(alpha = 0.5), max_samples=1.0, max_features=1.0, n_estimators=n_estimators, n_jobs = -1)
        model = MultinomialNB(alpha=alpha)
        model.fit(self.x_train, self.y_train)
        pred = model.predict(self.x_test)
        print(" MultinomialNB Regression : accurancy_is", metrics.accuracy_score(self.y_test, pred))

    def SelfNaiveByes(self):
        model = Berboulli_Naive_Bayes()
        model.train(self.x_train,self.y_train)
        print(model.score(self.x_test,self.y_test))

if __name__ == '__main__':

    data, label = mc(n_samples = 1000, n_features = 5, n_classes = 2, n_redundant = 0, n_repeated = 0, shift = 0)
    data = np.where(data > 0, 1, 0)
    label = np.where(label == 0, 1,2)
    X_train, X_test, y_train, y_test = Feature_Processer().split(data,label,0.9,True)
    print(X_train)

    clf = classifier(X_train, X_test, y_train, y_test)
    clf.multNB(0.2)
    clf.SelfNaiveByes()

            