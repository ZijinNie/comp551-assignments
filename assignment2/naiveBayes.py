# -*- coding: utf-8 -*-


import numpy as np

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
                    if (yk[l] == j+1) & (X[l,j] == 1):
                        num_j_k +=1
                        
                theta_j_k[k,j] = (num_j_k + 1) / (y_sum + 2)
                #print(theta_j_k[k,j])
            
            
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
                print(prob)
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
        return count/ len(X_test)
        

#TODO: Accurancy function needed.
            