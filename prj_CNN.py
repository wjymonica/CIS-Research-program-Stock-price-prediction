# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:52:08 2019

@author: Administrator
"""

from __future__ import print_function
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
###### ubuntu ######
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import time


###### windows ######
#from tensorflow.python import keras
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Dropout, Flatten
#from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.python.keras import backend as K

batch_size = 64
num_classes = 2
#num_classes = 3
epochs = 3

cols = 30
offsetTrain = 1000
numTrain = 10000    # [offsetTrain, offsetTrain+numTrain+m+n-1)
offsetTest = offsetTrain+numTrain 
numTest = 3000     # [offsetTest, offsetTest+numTest+m+n-1)


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

#preprocessing step 1 - price=(price-LowerLimitPrice)/(UpperLimitPrice-LowerLimitPrice)
ppVals = np.copy(srcVals)
ppVals[:,108] = (ppVals[:,108]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #midPrice
ppVals[:,111] = (ppVals[:,111]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #LastPrice
for priceCol in range(116,126):
    ppVals[:,priceCol] = (ppVals[:,priceCol]-ppVals[:,138])/(ppVals[:,137]-ppVals[:,138])  #AskPrice,BidPrice

#preprocessing step 2 - drop some columns
# 139-6=133, drop UpdateTime,UpdateMillisec,Volume,Turnover,UpperLimitPrice,LowerLimitPrice
ppVals = np.delete(ppVals, [109,110,112,114,137,138], 1)
ppVals= preprocessing.normalize(ppVals, norm='l2')
for n in[10,15]:
    for m in [2*n,4*n,6*n,8*n]:
#    for m in [2*n]:

        print ("(m,n)=",(m,n))
        img_rows = m
        img_cols = cols
        
        #preprocessing step 3 - scaler
        rowsTrain = numTrain+m+n-1
        rowsTest = numTest+m+n-1
        ppTrain = ppVals[offsetTrain:offsetTrain+rowsTrain,:]
        ppTest = ppVals[offsetTest:offsetTest+rowsTest,:]
        
        scaler = MinMaxScaler().fit(ppTrain)
        ppTrain = scaler.transform(ppTrain)
        ppTest = scaler.transform(ppTest)
        
        #preprocessing step 4 - PCA
        #pca = PCA(n_components=0.95, svd_solver='full')  # return 25-28 feathers, set to 30
        #print(pca.explained_variance_ratio_)
        pca = PCA(n_components=cols)
        pca.fit(ppTrain)
        ppTrain = pca.transform(ppTrain)
        ppTest = pca.transform(ppTest)
        
        #make training data
        x_train = np.zeros((1,m,cols))
        y_train = np.zeros((1,num_classes))
        index = 0
        for i in range(m-1, rowsTrain-n):
            x_index = np.reshape(ppTrain[i-m+1:i+1], (1,m,cols))
            if(srcVals[i+offsetTrain+n,108] < srcVals[i+offsetTrain,108]):    # compare midPrice
                y_index = [[0,1]]
        #        y_index = [[0,0,1]]
        #    elif(srcVals[i+offsetTrain+n,108] == srcVals[i+offsetTrain,108]):    # compare midPrice
        #        y_index = [[0,1,0]]
            else:
                y_index = [[1,0]]
        #        y_index = [[1,0,0]]
            
            if(index == 0):
                x_train = x_index
                y_train = y_index
            else:
                x_train = np.insert(x_train, index, x_index, axis=0)
                y_train = np.insert(y_train, index, y_index, axis=0)
            index = index + 1
        
        print(x_train.shape)    # (numTrain,m,cols)
        print(y_train.shape)    # (numTrain,num_classes)
        
        #make testing data
        x_test = np.zeros((1,m,cols))
        y_test = np.zeros((1,num_classes))
        index = 0
        for i in range(m-1, rowsTest-n):
            x_index = np.reshape(ppTest[i-m+1:i+1], (1,m,cols))
            if(srcVals[i+offsetTest+n,108] < srcVals[i+offsetTest,108]):
                y_index = [[0,1]]
        #        y_index = [[0,0,1]]
        #    elif(srcVals[i+offsetTest+n,108] == srcVals[i+offsetTest,108]):
        #        y_index = [[0,1,0]]
            else:
                y_index = [[1,0]]
        #        y_index = [[1,0,0]]
            
            if(index == 0):
                x_test = x_index
                y_test = y_index
            else:
                x_test = np.insert(x_test, index, x_index, axis=0)
                y_test = np.insert(y_test, index, y_index, axis=0)
            index = index + 1
        
#        print(x_test.shape)    # (numTest,m*cols)
#        print(y_test.shape)    # (numTest,num_classes)
        
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)
        
        time_start=time.time()

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  #steps_per_epoch=numTrain//batch_size,  # windows only
                  verbose=1,
                  validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
#        print('Test loss:', score[0])
 #       print('Test accuracy:', score[1])
        
        y_pred = model.predict(x_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        
        y_test_class = np.argmax(y_test, axis=1)
        
        print(confusion_matrix(y_test_class, y_pred_class))
        time_end=time.time()
        print('time cost',time_end-time_start,'s') 
