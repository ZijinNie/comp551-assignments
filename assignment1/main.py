# -*- coding: utf-8 -*-

import logistic_regression as lr
import numpy as np
import matplotlib.pyplot as plt
import math
import LDA
file_path1= "winequality-red.csv"
file_path2="breast-cancer-wisconsin.data"

def genDataWOHeader (file_path):
    data = np.genfromtxt(file_path, delimiter = ";" , skip_header = 1)
    return data
def genData (file_path):
    data = np.genfromtxt(file_path, delimiter = ",")
    return data

#categolize the last column of redwine data to 0 and 1
#@param data: dataset
def qualityToCategory (data):
    for i in range(data.shape[0]):
        if data[i,-1] >5:
            data[i,-1] = 1
        else:
            data[i,-1] = 0
            
#categorize the last column of cancer data to 0 and 1
#@param data: dataset
def classToCategory (data):
    for i in range(data.shape[0]):
        if data[i,-1] ==4:
            data[i,-1] = 1
        else:
            data[i,-1] = 0
    
#seperate test and train data 
#@param data: whole dataset
#@return testSet: testSet
#@return trainSet: trainSet
def seperateTestSet(data):
    dataSize= data.shape[0]
    testSize=math.floor(dataSize/10)
    testSet= data[:testSize, :]
    trainSet = data[testSize:, :]
    
    return testSet, trainSet

#remove the rows with 'nan' in the dataset
#@Param data: whole dataset
def preprocessData(data):
    
    data=data[~np.isnan(data).any(axis=1)]
    
    return data
            
# remove the dateset outside 3* standard deviation
# @Param data: dataset
#@return: cleaned data
def removeOutLiersByND (data):
    x = data[: , :-1]
    outLierList = np.array([])
    for i in range(x.shape[1]):
        mean = np.mean(x[: , i])
        deviation = np.std(x[: , i])
        for j in range(x.shape[0]):
            if x[j, i] >= (mean+ 3*deviation) or x[j, i] <= (mean- 3*deviation):
                np.append(outLierList, j)
    np.unique(outLierList)
    np.delete(data, outLierList, axis= 0)
    return data

#select subset of all features by choosing the best
#performing subset of all subsets given a model
#@param model: an instance of a model
#@Param data: whole dataset
#@return bestPerformingFeatures : the set of features that performs the best
def featureSelection (model, data):
    selectedFeatureNum = []
    selectedFeatureArray = -1 
    bestAccuracyAll = 0
    y_2d = np.array([data[:, -1]]).T
    #print(y_2d)
    for i in range(data.shape[1]-1):
        featureToAdd = -1
        bestAccuracy = 0
        column_2d = -1
        if i==0:
            
            for j in range(data.shape[1]-1):
                if (j in selectedFeatureNum ) == False:
                    column_2d = np.array([data[:, j]]).T
                    nums = selectedFeatureNum.append(j)
                    
                    # ------5 should be changed --
                    accuracy = kFoldValidation(model, np.concatenate((column_2d,y_2d), axis = 1),5)
                    print ("Using feature(s){} accuracy is".format(nums, accuracy))
                    if accuracy >= bestAccuracy :
                        bestAccuracy = accuracy
                        featureToAdd = j
            selectedFeatureArray = column_2d 
            bestAccuracyAll = bestAccuracy
            selectedFeatureNum.append(featureToAdd)
                                      
            continue
        else:   
            #try add feature from the rest of set
            for j in range(data.shape[1]-1):
                if (j in selectedFeatureNum ) == False:
                    column_2d = np.array([data[:, j]]).T
                    nums = selectedFeatureNum.append(j)
                    
                    # ------5 should be changed ---
                    print(selectedFeatureArray)
                    accuracy = kFoldValidation(model, np.concatenate((selectedFeatureArray, column_2d , y_2d), axis = 1),5)
                    print ("Using feature(s){} accuracy is".format(nums, accuracy))
                    if accuracy >= bestAccuracy :
                        bestAccuracy = accuracy
                        featureToAdd = j
        #additional feature cannot improve performance
        if bestAccuracyAll>= bestAccuracy:
            print ("maxima reached")
            break
        else:
            #add addtional feature
            bestAccuracyAll = bestAccuracy
            selectedFeatureNum.append(featureToAdd)
            selectedFeatureArray = np.concatenate((selectedFeatureArray,np.array([data[:, j]]).T),axis =1)
    print("feature selection ended, best performing features are {}, the accuracy is {}".format(selectedFeatureNum, bestAccuracyAll))
    return selectedFeatureNum, selectedFeatureArray
        

#normalize data to z-score by each feature except the last column
#@Param data: dataset 
#@return data: the normalized data set
def normalize(data):
    for i in range(data.shape[1]-1):
        mean = np.mean(data[: , i])
        deviation = np.std(data[: , i])
        for j in range(data.shape[0]):
            data[i,j] = (data[i,j] - mean)/deviation
    return data     

#do k fold validation
#@param model: the instance of the model
#@param trainData: the train data
#@param k: the number of folds for validation
#@return accuracy: the accuracy of the model on the given dataset
def kFoldValidation (model, trainData, k):
        size = int(len(trainData) / k)
        error = 0
        for i in range(k):
            if (i == k-1):
                validationSet = trainData[(i * size) : , :]
                trainSet = trainData[ : (i * size) , :]
            else:
                validationSet = trainData[(i * size): ((i + 1) * size), :]
                trainSet = np.concatenate((trainData[0 : i* size, :], trainData[(i + 1)* size : , :]),axis=0)
                
            x_train= trainSet[: , :-1]
            y_train= trainSet[:, -1]
            
            print(x_train.shape)
            p0, p1, u0, u1, covariance = model.fit(x_train, y_train)
            
            Y_predict = model.predict(validationSet[: , :-1],u0,u1,p0,p1,covariance)
            Y_true = validationSet[:, -1]
            count = 0
            for j in range(len(Y_true)):
                if Y_predict[j] != Y_true[j]:
                    count +=1
            print("count is{}".format(count ))
            error += count
            
        avg_error = error / k
        accuracy= 1- avg_error / (trainData.shape[0]/k)
        return accuracy
 
#initialize wine data
data1 = genDataWOHeader(file_path1)
qualityToCategory(data1)
#print(data1)

#initialize cancer data
data2 = genData(file_path2)
data2=preprocessData(data2)
classToCategory(data2)


np.random.shuffle(data1)
data1= removeOutLiersByND(data1)
testSet, trainSet = seperateTestSet(data1)
trainSet=np.insert(trainSet, trainSet.shape[1]-1,np.ones((trainSet.shape[0],1),dtype=float),axis=1)
aModel= lr.LogisticRegression(0.01,1000)
aModel.k_fold_train_and_validation(trainSet,5)
print("lr model created, step is {}, n_iteration is {}".format(aModel.step, aModel.n_iterations))



'''
#shuffle the data
np.random.shuffle(data1)
#data1= removeOutLiersByND(data2)
testSet, trainSet = seperateTestSet(data1)
aModel = LDA.LDA()
'''
#featureSelection(aModel,data1)
kFoldValidation(aModel,trainSet, 5)