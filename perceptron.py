#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:01:59 2018

@author: noch
"""

import numpy as np

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