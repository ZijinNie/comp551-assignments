# -*- coding: utf-8 -*-


import numpy as np

class Berboulli_Naive_Bayes:
    
    def train (self,x_train,y_train):
        X = x_train
        y = y_train
        
        #get the number of classes
        classNum = np.amax(y)
        
        # prob function = constant + prameter * x
        # function contains the prob function of all classes
        # class numebr start from 1 
        # the array has the following format: class1 [constant, parameter1,..., parameter m]
        function = np.ones( ( classNum, (X.shape[1]+1) ) )
        
        
        
        #for each class, train and find the log_odds
        for i in range(classNum):
            
            #array of theta j,1 and theta j,0. First col for 0, second for 1
            theta = np.zeros((X.shape[1],2))
           
            yi = np.array(y)
            
            #number of class 1 and 0
            n1 = 0
            n0 = 0
            
            #categorize the classes into binary
            for j in range(len(yi)):
                if yi[j] == (i+1):
                    yi[j] = 1
                    n1 +=1
                else:
                    yi[j] = 0
                    n0+=1
            #number of examples where xj =1 and y =1 
            numj1 = 0
            #number of examples where xj =1 and y =0 
            numj0 = 0
            for k in range(X.shape[1]):
                for l in range(X.shape[0]):
                    if X[l,k] == 1:
                        if y [l] == 1:
                            numj1 +=1
                        else:
                            numj0 +=1
                theta[k,0] = numj0 / n0
                theta[k,1] = numj1 / n1
            
            #calculate log( p1/ p0)
            logRatio = np.log( n1/n0 )
            
            constant = 0
            parameter = 0
            for m in range(X.shape[1]):
                constant += self.w0(theta,m)
                parameter += (self.w1(theta,m) - self.w0(theta,m))
                function[i,m+1] = parameter
            
            constant += logRatio
            function[i,0] = constant
            
            self.function = function
            
        return function
                
                
                
    def w0 (self, theta, j):
        return np.log( (1- theta[j,1]) / (1 - theta[j,0]))
    
    def w1 (self, theta, j):
        return np.log( theta[j , 1] - theta[j , 0])
        
    def fit (self, x):
        function = self.function                  
        bestClass = 0
        bestProb = 0
        
        #Choose the class with highest prob
        result = np.zeros((x.shape[0],1))
        index  = 0
        for x_i in x:
            for i in range(function.shape[0]):
                prob = function[x_i, 0]
                w = function [ 1 : ]
                prob +=np.dot( np.array(x_i.T) , w)
                if prob > bestProb:
                    bestClass = i+1
                    bestProb = prob
                    result[index] = bestClass
            index +=1
        return result

#TODO: Accurancy function needed.
            