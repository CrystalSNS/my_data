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
        l = np.dot(X_test[i],w) + b
        if l <=0 :
            Y_predicted.append(-1)
        else:
            Y_predicted.append(1)
    return Y_predicted

def test_with_id(w, b, X):
    
    #X = np.c_[ X, np.ones(X.shape[0])]
    result = pd.DataFrame(columns = ['Id','Bound'])
    X_test = pd.DataFrame.as_matrix(X.loc[:, X.columns != 'Id'])
    X_te_id = pd.DataFrame.as_matrix(X.loc[:,'Id'])
    
    for i, x in enumerate(X_test):
        l = np.dot(X_test[i],w) + b
        result.loc[i] = 0
        if l <=0 :
            result.loc[i]['Id'] = X_te_id[i]
            result.loc[i]['Bound'] = -1
        else:
            result.loc[i]['Id'] = X_te_id[i]
            result.loc[i]['Bound'] = 1
    return result

def test_ker_id(X_tr, Y_tr, X_te, alpha, b, z):
    
    #X_tr = np.c_[ X_tr, np.ones(X_tr.shape[0])]
    #X_te = np.c_[ X_te, np.ones(X_te.shape[0])]
    X_test = pd.DataFrame.as_matrix(X_te.loc[:, X_te.columns != 'Id'])
    X_train = pd.DataFrame.as_matrix(X_tr.loc[:, X_tr.columns != 'Id'])
    X_te_id = pd.DataFrame.as_matrix(X_te.loc[:,'Id'])
    result = pd.DataFrame(columns = ['Id','Bound'])
    
    for i, x_i in enumerate(X_test):
        sm = np.sum(alpha * Y_tr * kernel(X_train, x_i, z)) - b
        result.loc[i] = 0
        if sm <=0 :
            result.loc[i]['Id'] = X_te_id[i]
            result.loc[i]['Bound'] = -1
        else:
            result.loc[i]['Id'] = X_te_id[i]
            result.loc[i]['Bound'] = 1
    return result

def kernel(x, y, z):
    #degree-2 polynomials
    return (np.dot(x, y) + z)**2

def kernel_(x, y, sigma):

    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- np.linalg.norm(x - y) / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- np.linalg.norm(x - y, axis=1) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], axis=2) / (2 * sigma ** 2))
    return result

def test_ker(X_tr, Y_tr, X_te, alpha, b, z):
    
    #X_train = np.c_[X_tr, np.ones(X_tr.shape[0])]#add bias column of 1 to X_train
    #X_test = np.c_[X_te, np.ones(X_te.shape[0])]#add bias column of 1 to X_test
    X_train = X_tr
    X_test = X_te
    Y_predicted = []
    
    for i, x_i in enumerate(X_test):
        #sm = 0
        #for j, x_j in enumerate(X_train):
        #    sm = sm + (alpha[j] * Y_tr[j] * linear_kernel(x_j, x_i)) + b
        result = np.sum(alpha * Y_tr * kernel(X_train, x_i, z)) - b
        if result <=0 :
            Y_predicted.append(-1)
        else:
            Y_predicted.append(1)
    return Y_predicted






























