#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:10:03 2018

@author: noch
"""
import numpy as np
import pandas as pd

def test(w, b, X_test):
    Y_predicted = []
    for i, x in enumerate(X_test):
        if(np.dot(X_test[i],w) + b <=0 ):
            Y_predicted.append(-1)
        else:
            Y_predicted.append(1)
    return Y_predicted

def test_with_id(w, b, X):
    result = pd.DataFrame(columns = ['Id','Bound'])
    X_test = pd.DataFrame.as_matrix(X.loc[:, X.columns != 'Id'])
    Y_test = pd.DataFrame.as_matrix(X.loc[:,'Id'])
    for i, x in enumerate(X_test):
        result.loc[i] = 0
        if(np.dot(X_test[i],w) + b <=0 ):
            result.loc[i]['Id'] = Y_test[i]
            result.loc[i]['Bound'] = -1
        else:
            result.loc[i]['Id'] = Y_test[i]
            result.loc[i]['Bound'] = 1

    return result