# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 21:33:29 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import time


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
ppVals = np.copy(srcVals)
ppVals[:,108] = (ppVals[:,108]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #midPrice
ppVals[:,111] = (ppVals[:,111]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #LastPrice
for priceCol in range(116,126):
    ppVals[:,priceCol] = (ppVals[:,priceCol]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #AskPrice,BidPrice

#preprocessing step 2 - drop some columns
# 139-6=133, drop UpdateTime,UpdateMillisec,Volume,Turnover,UpperLimitPrice,LowerLimitPrice
ppVals = np.delete(ppVals, [109,110,112,114,137,138], 1)
#ppVals -= np.mean(ppVals, axis = 0) # 减去均值，使得以0为中心
#ppVals /= np.std(ppVals, axis = 0) # 归一化
#print(ppVals.shape)
ppVals= preprocessing.normalize(ppVals, norm='l2')
print(ppVals.shape)

for n in[5, 10, 15]:
    for m in [2*n, 4*n, 6*n, 8*n]:

        print ("(m,n)=",(m,n))
        #preprocessing step 3 - scaler
        rowsTrain = numTrain+m+n-1
        rowsTest = numTest+m+n-1
        ppTrain = ppVals[offsetTrain:offsetTrain+rowsTrain,:]
        ppTest = ppVals[offsetTest:offsetTest+rowsTest,:]
        
        scaler = MinMaxScaler().fit(ppTrain)
        ppTrain = scaler.transform(ppTrain)
        ppTest = scaler.transform(ppTest)
        
        #preprocessing step 4 - PCA
        #pca = PCA(n_components=0.95, svd_solver='full')  # return 25-28 feathers, set cols=30
       ·#print(pca.explained_variance_ratio_)
        pca = PCA(n_components=cols)
        pca.fit(ppTrain)
        ppTrain = pca.transform(ppTrain)
        ppTest = pca.transform(ppTest)
        
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
        
#        print(x_train.shape)    # (numTrain,m*cols)
#        print(y_train.shape)    # (numTrain,)
        
        #make testing data
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

        for c in [0.01,0.1,10,100]:
            time_start=time.time()
            lnSVC = LinearSVC(C=c).fit(x_train, y_train)
            score_train = lnSVC.score(x_train, y_train)
            score_test = lnSVC.score(x_test, y_test)
            print("LinearSVC, C=",c)
            print(score_train, score_test)
            print(confusion_matrix(y_test, lnSVC.predict(x_test)))
            time_end=time.time()
            print('time cost',time_end-time_start,'s') 
           
            time_start=time.time()
            svc = SVC(C=c).fit(x_train, y_train)
            score_train = svc.score(x_train, y_train)
            score_test = svc.score(x_test, y_test)
            print("SVC, C=",c)
            print(score_train, score_test)
            print(confusion_matrix(y_test, svc.predict(x_test)))
            time_end=time.time()
            print('time cost',time_end-time_start,'s')

        time_start=time.time()
        logReg = LogisticRegression().fit(x_train, y_train)
        score_train = logReg.score(x_train, y_train)
        score_test = logReg.score(x_test, y_test)
        print("LogisticRegression")
        print(score_train, score_test)
        print(confusion_matrix(y_test, logReg.predict(x_test)))
        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        
        time_start=time.time()
        lnSVC = LinearSVC().fit(x_train, y_train)
        score_train = lnSVC.score(x_train, y_train)
        score_test = lnSVC.score(x_test, y_test)
        print("LinearSVC")
        print(score_train, score_test)
        print(confusion_matrix(y_test, lnSVC.predict(x_test)))
        time_end=time.time()
        print('time cost',time_end-time_start,'s') 
       
        time_start=time.time()
        svc = SVC().fit(x_train, y_train)
        score_train = svc.score(x_train, y_train)
        score_test = svc.score(x_test, y_test)
        print("SVC")
        print(score_train, score_test)
        print(confusion_matrix(y_test, svc.predict(x_test)))
        time_end=time.time()
        print('time cost',time_end-time_start,'s')
        
        time_start=time.time()
        bnl = BernoulliNB().fit(x_train, y_train)
        score_train = bnl.score(x_train, y_train)
        score_test = bnl.score(x_test, y_test)
        print("BernoulliNB")
        print(score_train, score_test)
        print(confusion_matrix(y_test, bnl.predict(x_test)))
        time_end=time.time()
        print('time cost',time_end-time_start,'s')