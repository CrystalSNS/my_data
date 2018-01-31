#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:22:53 2018

@author: noch
"""

import numpy as np
import cvxopt
#from cvxopt import matrix, solvers 
#from cvxopt.solvers import qp


def svm_f(X_tr, Y_tr, C):
    
    n_examples = X_tr.shape[0]
    
    #gram matrix of the examples
    K = np.zeros((n_examples,n_examples))
    
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i,j] = np.dot(X_tr[i],X_tr[j])
    
    #matrix computation for solver : quadratic programming            
    P = cvxopt.matrix(np.outer(Y_tr,Y_tr)*K)
    
    q = cvxopt.matrix(-1*np.ones((n_examples,1)))
    
    A = cvxopt.matrix(np.array(Y_tr),(1,n_examples))

    b = cvxopt.matrix(0.0)
    
    G1 = np.diag((-np.ones((n_examples))))
    h1 = np.zeros((n_examples,1))
    G2 = np.diag((np.ones((n_examples))))
    h2 = C*np.ones((n_examples,1))
    G = cvxopt.matrix(np.vstack((G1,G2)))
    h = cvxopt.matrix(np.vstack((h1,h2)))
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P,q,G,h,A,b)
    
    alphas = np.ravel(sol['x'])
    
    #####deduce weights from alpha
    w = np.zeros(X_tr.shape[1])
    for i in range(n_examples):
        w = w + alphas[i] * Y_tr[i] * X_tr[i]
    
    bias = 0
    nb_bias = 0
    for i in range(n_examples):
        if (alphas[i]> C/10000 and alphas[i]+C/1000 < C):
            bias = bias + Y_tr[i] - np.dot(X_tr[i], w)
            nb_bias = nb_bias + 1
    if nb_bias != 0:
        bias = bias/nb_bias
    return w, bias
    
