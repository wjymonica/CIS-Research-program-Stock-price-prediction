# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:50:10 2019

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import LSTM,Dense
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import LSTM,Dense

from sklearn.metrics import confusion_matrix

cols = 16    # keep cols features after PCA
offsetTrain = 1000    # begin position of training dataset
numTrain = 10000    # [offsetTrain, offsetTrain+numTrain+m+n-1) -> x_train(numTrain, m*cols)
offsetTest = offsetTrain+numTrain     # begin position of testing dataset
numTest = 3000     # [offsetTest, offsetTest+numTest+m+n-1) -> t_test(numTest, m*cols)
numClass = 2       # <,>= or <,=,>
batchSize = 32     # batch size in training
numEpochs = 3

# load data
def TimeStr2Num(s):
    s = s.decode()
    arTest = s.split(':')
    if(len(arTest) == 3):
        return int(arTest[0])*10000 + int(arTest[1])*100 + int(arTest[2])
    else:
        return 0

srcVals=np.loadtxt('dataPart.csv', delimiter=',', skiprows=1, converters={109:TimeStr2Num})

ppVals = np.copy(srcVals)
ppVals[:,108] = (ppVals[:,108]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #midPrice
ppVals[:,111] = (ppVals[:,111]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #LastPrice
for priceCol in range(116,126):
    ppVals[:,priceCol] = (ppVals[:,priceCol]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #AskPrice,BidPrice

#preprocessing step 2 - drop some columns
# 139-6=133, drop UpdateTime,UpdateMillisec,Volume,Turnover,UpperLimitPrice,LowerLimitPrice
ppVals = np.delete(ppVals, [109,110,112,114,137,138], 1)

for n in[5,10,15]:
    for m in [2*n,4*n,6*n,8*n]:
#    for m in [2*n]:
        print ("(m,n)=",(m,n))
        #make training data
        
        #preprocessing step 3 - scaler
        rowsTrain = numTrain+m+n-1
        rowsTest = numTest+m+n-1
        ppTrain = ppVals[offsetTrain:offsetTrain+rowsTrain,:]
        ppTest = ppVals[offsetTest:offsetTest+rowsTest,:]
        
        scaler = MinMaxScaler().fit(ppTrain)
        ppTrain = scaler.transform(ppTrain)
        ppTest = scaler.transform(ppTest)
        
        #preprocessing step 4 - PCA
        #pca = PCA(n_components=0.95, svd_solver='full')  # return 25-28 feathers
        #print(pca.explained_variance_ratio_)
        pca = PCA(n_components=cols)
        pca.fit(ppTrain)
        ppTrain = pca.transform(ppTrain)
        ppTest = pca.transform(ppTest)

        x_train = np.zeros((1,m,cols))
        y_train = np.zeros((1,numClass))
        index = 0
        for i in range(m-1, rowsTrain-n):
            x_index = np.reshape(ppTrain[i-m+1:i+1], (1,m,cols))
            if(numClass == 2):
                if(srcVals[i+offsetTrain+n,108] < srcVals[i+offsetTrain,108]):
                    y_index = [[0,1]]
                else:
                    y_index = [[1,0]]
            else:
                if(srcVals[i+offsetTrain+n,108] < srcVals[i+offsetTrain,108]):
                    y_index = [[0,0,1]]
                if(srcVals[i+offsetTrain+n,108] == srcVals[i+offsetTrain,108]):
                    y_index = [[0,1,0]]
                else:
                    y_index = [[1,0,0]]
        
            if(index == 0):
                x_train = x_index
                y_train = y_index
            else:
                x_train = np.insert(x_train, index, x_index, axis=0)
                y_train = np.insert(y_train, index, y_index, axis=0)
            index = index + 1
        
        print(x_train.shape)
        print(y_train.shape)
        
        #make testing data
        x_test = np.zeros((1,m,cols))
        y_test = np.zeros((1,numClass))
        index = 0
        for i in range(m-1, rowsTest-n):
            x_index = np.reshape(ppTest[i-m+1:i+1], (1,m,cols))
            if(numClass == 2):
                if(srcVals[i+offsetTest+n,108] < srcVals[i+offsetTest,108]):
                    y_index = [[0,1]]
                else:
                    y_index = [[1,0]]
            else:
                if(srcVals[i+offsetTest+n,108] < srcVals[i+offsetTest,108]):
                    y_index = [[0,0,1]]
                elif(srcVals[i+offsetTest+n,108] == srcVals[i+offsetTest,108]):
                    y_index = [[0,1,0]]
                else:
                    y_index = [[1,0,0]]
            
            if(index == 0):
                x_test = x_index
                y_test = y_index
            else:
                x_test = np.insert(x_test, index, x_index, axis=0)
                y_test = np.insert(y_test, index, y_index, axis=0)
            index = index + 1
        
        print(x_test.shape)
        print(y_test.shape)
        
        model = Sequential()
        
        model.add(LSTM(32, return_sequences=True, input_shape=(m, cols)))
        model.add(LSTM(32, return_sequences=True))
        model.add(LSTM(32))
        
        model.add(Dense(numClass, activation='softmax'))
        
        model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        
        model.fit(x_train, y_train, batch_size=batchSize, epochs=numEpochs, validation_data=(x_test,y_test))
        
        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        
        y_pred = model.predict(x_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        y_test_class = np.argmax(y_test, axis=1)
        
        print(confusion_matrix(y_test_class, y_pred_class))
