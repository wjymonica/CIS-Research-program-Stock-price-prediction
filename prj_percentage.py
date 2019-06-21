# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# load data
def TimeStr2Num(s):
    s = s.decode()
    arTest = s.split(':')
    if(len(arTest) == 3):
        return int(arTest[0])*10000 + int(arTest[1])*100 + int(arTest[2])
    else:
        return 0

srcVals=np.loadtxt('dataPart.csv', delimiter=',', skiprows=1, converters={109:TimeStr2Num})
#srcVals=np.loadtxt('G:\JI\VE445_CS229\VE445\Project\\dataSet2.csv', delimiter=',', skiprows=0, converters={109:TimeStr2Num})

#srcVals = srcVals[::10, :]

offsetTrain = 1000    # begin position of training dataset
numTrain = 10000    # [offsetTrain, offsetTrain+numTrain+m+n-1) -> x_train(numTrain, m*cols)
offsetTest = offsetTrain+numTrain     # begin position of testing dataset
numTest = 3000     # [offsetTest, offsetTest+numTest+m+n-1) -> t_test(numTest, m*cols)

for n in[5, 10, 15]:
    for m in [2*n, 4*n, 6*n, 8*n]:
        print ("(m,n)=",(m,n))

        rowsTrain = numTrain+m+n-1
        rowsTest = numTest+m+n-1
        
        num0 = 0
        num1 = 0
        
        for i in range(m-1, rowsTrain-n):
            if(srcVals[i+offsetTrain+n,108] < srcVals[i+offsetTrain,108]):
                num1 = num1 + 1
            else:
                num0 = num0 + 1
                
        print(num0, num1, num0+num1, num0*1.0/(num0+num1))
        
        num0 = 0
        num1 = 0
        
        for i in range(m-1, rowsTest-n):
            if(srcVals[i+offsetTest+n,108] < srcVals[i+offsetTest,108]):
                num1 = num1 + 1
            else:
                num0 = num0 + 1
                
        print(num0, num1, num0+num1, num0*1.0/(num0+num1))