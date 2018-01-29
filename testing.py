#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:10:03 2018

@author: noch
"""
import numpy as np
import pandas as pd

def test(w, b, X_test):
    X_test = np.c_[ X_test, np.ones(X_test.shape[0])]
    Y_predicted = []
    for i, x in enumerate(X_test):
        #l = np.dot(X_test[i],w) + b
        print(np.dot(X_test[i],w))
        l = (np.dot(X_test[i],w) + 1)**2         
        if l <=0 :
            Y_predicted.append(-1)
        else:
            Y_predicted.append(1)
    return Y_predicted

def test_with_id(w, b, X):
    X = np.c_[ X, np.ones(X.shape[0])]
    result = pd.DataFrame(columns = ['Id','Bound'])
    X_test = pd.DataFrame.as_matrix(X.loc[:, X.columns != 'Id'])
    Y_test = pd.DataFrame.as_matrix(X.loc[:,'Id'])
    for i, x in enumerate(X_test):
        #l = np.dot(X_test[i],w) + b
        l = (np.dot(X_test[i],w) + 1)**2  
        result.loc[i] = 0
        if l <=0 :
            result.loc[i]['Id'] = Y_test[i]
            result.loc[i]['Bound'] = -1
        else:
            result.loc[i]['Id'] = Y_test[i]
            result.loc[i]['Bound'] = 1
    return result