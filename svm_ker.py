#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:19:26 2018

@author: noch
"""


import numpy as np
import cvxopt
#from cvxopt import matrix, solvers 
#from cvxopt.solvers import qp
def kernel_(x, y, z):
    #degree-2 polynomials
    return (np.dot(x, y) + z)**2

def kernel(x, y, sigma):

    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- np.linalg.norm(x - y) / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- np.linalg.norm(x - y, axis=1) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], axis=2) / (2 * sigma ** 2))
    return result
"""Returns the gaussian similarity of arrays `x` and `y` with
kernel width parameter `sigma` (set to 1 by default)."""
                 

def svm_f(X_tr, Y_tr, C, z):
    
    n_examples = X_tr.shape[0]
    
    #gram matrix of the examples
    K = np.zeros((n_examples,n_examples))
    
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i,j] = kernel(X_tr[i],X_tr[j], z)
    
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
    
    nb_bias = 0
    bias = 0
    for i in range(n_examples):
        if (alphas[i]> C/10000 and alphas[i]+C/1000 < C):
            bias = bias + np.sum(alphas * Y_tr * kernel(X_tr, X_tr[i], z)) - Y_tr[i]
            nb_bias = nb_bias + 1
        if (nb_bias == 5):
            break
    if nb_bias != 0:
        bias = bias/nb_bias
        
    return alphas, bias
    



























