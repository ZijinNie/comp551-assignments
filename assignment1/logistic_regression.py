# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:45:25 2019

"""

import numpy as np
import math

class LogisticRegression :
    step = 0
    n_iterations = 0
    
    def __init__ (self, step, n_iterations):
        self.step=step
        self.n_iterations=n_iterations
    
    def logistic(self,w,Xi):
        """ Compute the logistic function, given w and the features """
        # print(w.T)
        # print(Xi)
        a = np.dot(w.T,Xi)
        return 1/(1+np.exp(-a))
    
    
    #x: traning data
    #y: traning resut
    #w: parameters
    #m: size of training set
    
    def fit(self, x, y):
        '''
        n = x.shape[0]
        x_dummy = np.ones((n,1))
        w = np.random.normal(0,1,x.shape[1]+1)
        x= np.concatenate((x, x_dummy),axis=1)
        
        
        for _ in range(self.n_iterations):
            
            gradient = np.zeros(x.shape[1])
            
            for i in range (n):
                log_odds= np.dot(w.T,x[i,:])
                logistic_function= 1/(1+np.exp(-log_odds))
                gradient= np.add(gradient, np.dot(x[i,:],(y[i]- logistic_function)))
                
            w=np.add(w, (self.step)* gradient)
        print(w.shape)
        
        return w 
        #for _ in range(n) : 
         #   log_odds= np.dot(self.w.T,x[i,:])
          #  logistic_function= 1/(1+np.exp(-log_odds))
           # self.error -= (y[i]*math.log(logistic_function)+(1-y[i])*math.log(1-logistic_function))
        '''
        n = x.shape[0]
        m = x.shape[1]
       
        w = np.random.rand(m)
        ak = self.step
        count = 0 
        for i in range(m):
            x[:,i] = np.divide(x[:,i] - np.min(x[:,i]),np.max(x[:,i]) - np.min(x[:,i]))
        for _ in range(500):
            der = np.zeros((m))
            for i in range(n):
              der += (y[i] - self.logistic(w,x[i,:]))*x[i,:]
            w = w + ak * der
            count += 1 
        return w
        
    def predict(self, x, w):
        n=x.shape[0]        
        y= np.ones(n)
        for i in range(n):
            log_odds= np.dot(w.T,x[i,:])
            logistic_function= 1/(1+np.exp(-log_odds))
            if logistic_function >= 0.5:
                y[i] = 1
            else: 
                y[i] = 0
        return y
    
if __name__ == "__main__":
    

    
    file_path= "winequality-red.csv"
    
    def genDataWOHeader (file_path):
        data = np.genfromtxt(file_path, delimiter = ";",
                          skip_header = 1)
        return data
    
    def qualityToCategory (data):
        for i in range(data.shape[0]):
            if data[i,-1] >5:
                data[i,-1] = 1
            else:
                data[i,-1] = 0
    
    def seperateTestSet(data):
        dataSize= data.shape[0]
        testSize=math.floor(dataSize/10)
        testSet= data[:testSize, :]
        trainSet = data[testSize:, :]
        return testSet, trainSet
    
    ''' k: the number of fold
        data: the train set
    '''
    #def kFoldValidation(data,k):
        
    def seperateByK(data,k):
        n = data.shape[0]
        size = math.floor(n/k)
        segments = []
        for i in range(k):
            lower = math.floor( i *size)
            upper = math.floor( (i+1)*size)
            segments.append(data[lower:upper, :] )
        return segments
    
    def normalize(data):
        m = data.shape[1]
        for i in range(m):
                data[:,i] = np.divide(data[:,i] - np.min(data[:,i]),np.max(data[:,i]) - np.min(data[:,i]))
                
    def k_fold_train_and_validation(data, k, model):
        
        size = int(len(data) / k)
        error = 0
        for i in range(k):
            if (i == k-1):
                validationSet = data[(i * size) : , :]
                trainSet = data[ : (i * size) , :]
            else:
                validationSet = data[(i * size): ((i + 1) * size), :]
                trainSet = np.concatenate((data[0 : i* size, :], data[(i + 1)* size : , :]),axis=0)
            x_train= trainSet[: , :-1]
            x_validation = validationSet [:, :-1]
            y_train= trainSet[:, -1]
            
            normalize(x_train)
            normalize(x_validation)
          #  for i in range(x_train.shape[1]):
           #     x_train[:,i] = np.divide(x_train[:,i] - np.min(x_train[:,i]),np.max(x_train[:,i]) - np.min(x_train[:,i]))
            #    x_validation[:,i] = np.divide(x_validation[:,i] - np.min(x_validation[:,i]),np.max(x_validation[:,i]) - np.min(x_validation[:,i]))
            w= model.fit(x_train,y_train)
            Y_predict = model.predict(x_validation,w)
            Y_true = validationSet[:, -1]
            count = 0
            for j in range(len(Y_true)):
                if Y_predict[j] != Y_true[j]:
                    count +=1
            print("count is{}".format(count ))
            error += count
            
        avg_error = error / k
        print("# of data is {}".format(data.shape[0]))
        error_percentage= avg_error / (data.shape[0]/k)
        print(error_percentage)
        
        '''
        w= self.fit(data[:, :-1], data[:, -1])
        Y_predict = self.predict(data[:, :-1],w)
        Y_true = data[:, -1]
        count = 0
        for j in range(len(Y_true)):
            if Y_predict[j] != Y_true[j]:
                count +=1
        print("count is{}".format(count ))
        '''
        return count/ data.shape[0]
    data = genDataWOHeader(file_path)
    qualityToCategory(data)
    testSet, trainSet = seperateTestSet(data)
    aModel= LogisticRegression(0.01,500)
    print("lr model created, step is {}, n_iteration is {}".format(aModel.step, aModel.n_iterations))
    #trainSet=np.insert(trainSet, trainSet.shape[1]-1,np.ones((trainSet.shape[0],1),dtype=float),axis=1)
    np.random.shuffle(data)
    #aModel.fit(x,y)
    k_fold_train_and_validation(trainSet, 5,aModel)
    
    #print ("model error is {}, model w is {}".format(aModel.error,aModel.w))
        
        
        