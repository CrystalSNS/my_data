#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:08:19 2018

@author: michi
"""
from math import e
import numpy as np
import math
import pandas as pd


def sigmoid(scores):
    return 1 / (1 + np.e**(-scores))

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll


def logistic_regression(features, target, num_steps, learning_rate, add_intercept = True):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights = weights + learning_rate * gradient
    return weights

def test(p):
        l=0.0
        for i in range(0,p.shape[0]):
            l = l + p[i]
        l = l/p.shape[0]
        
        Y_predicted = np.zeros(p.shape[0])
        for i in range(0,p.shape[0]):
            if p[i] <= l:
                Y_predicted[i] = -1
            else:
                Y_predicted[i] = 1
        return Y_predicted
    
def test_with_id(p, X_te_id):
    
    result = pd.DataFrame(columns = ['Id','Bound'])

    l=0.0
    for i in range(0,p.shape[0]):
        l = l + p[i]
    l = l/p.shape[0]
    
    for i in range(0,p.shape[0]):
        result.loc[i] = 0
        if p[i] <= l:
            result.loc[i]['Id'] = X_te_id[i]
            result.loc[i]['Bound'] = -1
        else:
            result.loc[i]['Id'] = X_te_id[i]
            result.loc[i]['Bound'] = 1
    
    return result


def predict(X, p):
        intercept = np.ones((X.shape[0], 1))
        data_with_intercept = np.hstack((intercept,X))
        final_scores = np.dot(data_with_intercept,p)
        p_sig = sigmoid(final_scores)
        return p_sig


