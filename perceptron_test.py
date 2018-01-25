#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:55:40 2018

@author: noch
"""

import numpy as np
import pandas as pd
from random import randint
from itertools import product
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

def read_Xte0():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/testing_set/Xte0.csv', index_col=False, ).as_matrix()

def read_Xte1():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/testing_set/Xte1.csv', index_col=False).as_matrix()

def read_Xte2():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/testing_set/Xte2.csv', index_col=False).as_matrix()

def read_Xtr0():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Xtr0.csv', index_col=False, dtype={'DNA sequence': object})

def read_Xtr1():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Xtr1.csv', index_col=False)

def read_Xtr2():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Xtr2.csv', index_col=False)

def read_Ytr0():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Ytr0.csv', index_col=False)

def read_Ytr1():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Ytr1.csv', index_col=False)

def read_Ytr2():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Ytr2.csv', index_col=False)



def creat_col_name(r):
    permu = product("ATGC", repeat = r)
    arr = []
    for val in permu:
        st = ""
        for j in range(r):
            st = st + val[j]
        arr.append(st)
    return arr

#prepare data one char
def prepare_data(X, num_char):

    df = pd.DataFrame(columns = creat_col_name(num_char))
    col_name = list(df)

    for index, row in X.iterrows():
        ln = len(row['DNA'])
        df.loc[index] = 0
        for i in range(ln-num_char+1):
            s = str(row['DNA'])
            for n in col_name:
                st = ""
                for t in range(num_char):
                    st = st + s[i+t]
                    
                if(st == n):
                    df.loc[index][n] = df.loc[index][n]+1
                    break
                        
        for n in col_name:
            df.loc[index][n] = df.loc[index][n]/(ln-num_char+1)
                
        #if (index == 50):
        #   break     
    
    return df    
        
    #print(df)

def split_data(df, tr_num):
    msk = np.random.rand(len(df)) < (tr_num/100)
    return (df[msk], df[~msk])


def perceptron(X, Y):

    w = np.zeros(len(X[0]))
    eta = 0.05
    epoch = 1000
    b=0
    for t in range(epoch):
        for i, x in enumerate(X):
            if ((np.dot(X[i], w)+b)*Y[i]) <= 0:
                w = w + eta*X[i]*Y[i]
                b = b + eta*Y[i]
    return w, b

  
def test(w, b, X_test):
    Y_predicted = []
    for i, x in enumerate(X_test):
        if(np.dot(X_test[i],w) + b <=0 ):
            Y_predicted.append(-1)
        else:
            Y_predicted.append(1)
    return Y_predicted

#---------

X = read_Xtr0()
Y = read_Ytr0()
Y[Y['Bound'] == 0] = -1
 
for k in range(4):
    data_new = prepare_data(X, k+1)

    data_new['Bound'] = Y['Bound']
    
    data_train,  data_test = split_data(data_new, 70)
    
    tr_X = pd.DataFrame.as_matrix(data_train.iloc[:,:-1])
    tr_Y = pd.DataFrame.as_matrix(data_train['Bound'])
    
    te_X = pd.DataFrame.as_matrix(data_test.iloc[:,:-1])
    te_Y = pd.DataFrame.as_matrix(data_test['Bound'])
    
    w, b = perceptron(tr_X, tr_Y)
    
    Y_predicted_tr = test(w, b, tr_X)
    
    Y_predicted_te = test(w, b, te_X)
    
    print("\n \n Result_tr:" 
          + str(accuracy_score(Y_predicted_tr, tr_Y, normalize=False)) + 
          "/" + str(len(Y_predicted_tr)) 
          + "=" + str(accuracy_score(Y_predicted_tr, tr_Y, normalize=False)/len(Y_predicted_tr)))
    
    print("\n \n Result_te:" 
          + str(accuracy_score(Y_predicted_te, te_Y, normalize=False)) + 
          "/" + str(len(Y_predicted_te))
          + "=" + str(accuracy_score(Y_predicted_te, te_Y, normalize=False)/len(Y_predicted_te)))
    










