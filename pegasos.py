#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:59:57 2018

@author: noch
"""
import numpy as np
from random import randint

def pegasos(X, Y):
    
    eta = 0
    lmda = 0.0005
    epoch = 1000
    w = np.zeros(len(X[0]))
    S_p =  []
    Y_p =  []
    b = 0

    for t in range(1,epoch):
        
        for i, x in enumerate(X):
            if ((np.dot(X[i], w) + b)*Y[i]) < 1:
                S_p.append(X[i])
                Y_p.append(Y[i])
                
        eta = 1/(lmda * t)
        sm = 0
        sm_y = 0
        for idx, row in enumerate(S_p):
            sm = sm + (S_p[idx]*Y_p[idx])
            sm_y = sm_y + Y_p[idx]
        
        w = np.dot((1 - (lmda*eta)),w) + ((eta/X.shape[0]) * sm)
        b = (b*(1-(lmda*eta))) + ((eta/X.shape[0])*sm_y)
        
    return w, b 


def pegasos_(X, Y, lmda, epoch):
    
    eta = 0
    #lmda = 0.0005
    #epoch = 1000
    w = np.zeros(len(X[0]))
    b = 0

    for t in range(1,epoch):
        eta = 1/(lmda * t)
        
        i = randint(0, X.shape[0]-1)
        
        
        if ((np.dot(X[i], w) + b)*Y[i]) < 1:

            w = np.dot((1 - (lmda*eta)),w) + np.dot((eta*Y[i]),X[i])
            
            b = (b*(1-(lmda*eta))) + eta*Y[i]
        
        elif ((np.dot(X[i], w) + b)*Y[i]) >= 1:    
            w = np.dot((1 - (lmda*eta)),w)
            b = b*(1-(lmda*eta))
        
        if(np.linalg.norm(w) != 0):
            w = min(1,( 1/np.sqrt(lmda) )/ np.linalg.norm(w) ) * w
        
    return w, b 