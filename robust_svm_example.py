#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:47:04 2018

@author: noch
"""

from robsvm import robsvm
from cvxopt import matrix, normal, uniform

# parameters
m, n = 60, 2
gamma = 10.0

# generate random problem data
X = 2.0*uniform(m,n)-1.0
d = matrix(1,(m,1))

# generate noisy labels
w0 = matrix([2.0,1.0])+normal(2,1); b0 = 0.4
z = 0.2*normal(m,1)
for i in range(m):
    if (X[i,:]*w0)[0] + b0 < z[i]: d[i] = -1

# generate uncertainty ellipsoids
k = 2
P = [0.1*normal(4*n,n) for i in range(k)]
P = [ p.T*p for p in P]
e = matrix(0,(m,1))
for i in range(m):
    if d[i] == -1: e[i] = 1

# solve SVM training problem
w, b, u, v, iterations = robsvm(X, d, gamma, P, e)