#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:56:20 2019

@author: starmoon
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import time


m = 40    # m records as a input data
n = 10     # predict price rise and fall after n time stamps
cols = 30    # keep cols features after PCA
offsetTrain = 1000    # begin position of training dataset
numTrain = 10000    # [offsetTrain, offsetTrain+numTrain+m+n-1) -> x_train(numTrain, m*cols)
offsetTest = offsetTrain+numTrain     # begin position of testing dataset
numTest = 3000     # [offsetTest, offsetTest+numTest+m+n-1) -> t_test(numTest, m*cols)

# load data
def TimeStr2Num(s):
    s = s.decode()
    arTest = s.split(':')
    if(len(arTest) == 3):
        return int(arTest[0])*10000 + int(arTest[1])*100 + int(arTest[2])
    else:
        return 0

srcVals=np.loadtxt('dataPart.csv', delimiter=',', skiprows=1, converters={109:TimeStr2Num})
#srcVals=np.loadtxt('dataSet2.csv', delimiter=',', skiprows=0, converters={109:TimeStr2Num})

#plot midPrice vs index
#mp = srcVals[:,108]
#plt.plot(mp)

#preprocessing step 1 - price=(price-LowerLimitPrice)/(UpperLimitPrice-LowerLimitPrice)
ppVals = srcVals
ppVals[:,108] = (ppVals[:,108]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #midPrice
ppVals[:,111] = (ppVals[:,111]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #LastPrice
for priceCol in range(116,126):
    ppVals[:,priceCol] = (ppVals[:,priceCol]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #AskPrice,BidPrice

#preprocessing step 2 - drop some columns
# 139-6=133, drop UpdateTime,UpdateMillisec,Volume,Turnover,UpperLimitPrice,LowerLimitPrice
ppVals = np.delete(ppVals, [109,110,112,114,137,138], 1)



print ("(m,n)=",(m,n))
#preprocessing step 3 - scaler
rowsTrain = numTrain+m+n-1
rowsTest = numTest+m+n-1
ppTrain = ppVals[offsetTrain:offsetTrain+rowsTrain,:]

scaler = MinMaxScaler().fit(ppTrain)
ppTrain = scaler.transform(ppTrain)

#preprocessing step 4 - PCA
#pca = PCA(n_components=0.95, svd_solver='full')  # return 25-28 feathers, set cols=30
#print(pca.explained_variance_ratio_)
pca = PCA(n_components=cols)
pca.fit(ppTrain)
ppTrain = pca.transform(ppTrain)

#make training data
x_train = np.zeros((1,m*cols))
y_train = np.zeros((1,1))
index = 0
for i in range(m-1, rowsTrain-n):
    x_index = np.reshape(ppTrain[i-m+1:i+1], (1,m*cols))
    if(srcVals[i+offsetTrain+n,108] < srcVals[i+offsetTrain,108]):    # compare midPrice
#    if(srcVals[i+offsetTrain+n,108] == srcVals[i+offsetTrain,108]):    # compare midPrice
        y_index = [1]
    else:
        y_index = [-1]
    
    if(index == 0):
        x_train = x_index
        y_train = y_index
    else:
        x_train = np.insert(x_train, index, x_index, axis=0)
        y_train = np.insert(y_train, index, y_index, axis=0)
    index = index + 1
    
#train
logReg = LogisticRegression().fit(x_train, y_train)
score_train = logReg.score(x_train, y_train)

#        print(x_train.shape)    # (numTrain,m*cols)
#        print(y_train.shape)    # (numTrain,)

ArrAccu=[]
bestAccu=0
bestID=0
#make testing data
for i in range (250):
    print(i)
    ppTest = ppVals[offsetTest:offsetTest+rowsTest,:]
    ppTest = scaler.transform(ppTest)
    ppTest = pca.transform(ppTest)
    x_test = np.zeros((1,m*cols))
    y_test = np.zeros((1,1))
    index = 0
    for i in range(m-1, rowsTest-n):
        x_index = np.reshape(ppTest[i-m+1:i+1], (1,m*cols))
        if(srcVals[i+offsetTest+n,108] < srcVals[i+offsetTest,108]):
     #   if(srcVals[i+offsetTest+n,108] == srcVals[i+offsetTest,108]):
            y_index = [1]
        else:
            y_index = [-1]
        
        if(index == 0):
            x_test = x_index
            y_test = y_index
        else:
            x_test = np.insert(x_test, index, x_index, axis=0)
            y_test = np.insert(y_test, index, y_index, axis=0)
        index = index + 1
    
    #        print(x_test.shape)    # (numTest,m*cols)
    #        print(y_test.shape)    # (numTest,)
    
    score_test = logReg.score(x_test, y_test)
    ArrAccu.append(score_test)
    if(score_test>bestAccu):
        bestAccu=score_test
        bestID=i
    print(score_train, score_test)
    print(confusion_matrix(y_test, logReg.predict(x_test)))
    offsetTest+=300

plt.plot(ArrAccu)

