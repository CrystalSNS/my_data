#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:45:12 2018

@author: noch
"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler


class SMO_md:
    
    def __init__(self, X, Y, C, Alpha, Error, b, m):
        self.X = X               
        self.Y = Y              
        self.C = C               
        self.Alpha = Alpha    
        self.b = b               
        self.Error = Error     
        self._obj = []          
        self.m = len(self.X)    
        
def kernel_(x, y, b=1):
    #linear_kernel
    return np.dot(x, y.T) + b 
                           
def kernel(x, y, sigma=1):
    #gaussian_kernel
    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- np.linalg.norm(x - y) / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- np.linalg.norm(x - y, axis=1) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], axis=2) / (2 * sigma ** 2))
    return result

def obj_func(X, Y, Alpha):
    
    return np.sum(Alpha) - 0.5 * np.sum(Y * Y * kernel(X, X) * Alpha * Alpha)                               

def decision_func(X_tr, Y_tr, Alpha, b, X_te):
    kernel(X_tr, X_te)
    result = (Alpha * Y_tr) @ kernel(X_tr, X_te) - b
    
    return result

def plot_decision_boundary(md, ax, resolution=100, colors=('b', 'k', 'r')):

        xrange = np.linspace(md.X[:,0].min(), md.X[:,0].max(), resolution)
        yrange = np.linspace(md.X[:,1].min(), md.X[:,1].max(), resolution)
        grid = [[decision_func(md.X,
                               md.Y, 
                               md.Alpha,
                               md.b,
                               np.array([xr, yr])) for yr in yrange] for xr in xrange]
        grid = np.array(grid).reshape(len(xrange), len(yrange))
        
        ax.contour(xrange, yrange, grid, (-1, 0, 1), linewidths=(1, 1, 1),
                   linestyles=('--', '-', '--'), colors=colors)
        ax.scatter(md.X[:,0], md.X[:,1],
                   c=md.Y, cmap=plt.cm.viridis, lw=0, alpha=0.5)
        
        mask = md.Alpha != 0.0
        ax.scatter(md.X[:,0][mask], md.X[:,1][mask],
                   c=md.Y[mask], cmap=plt.cm.viridis)
        
        return grid, ax
    
def takeStep(i, j, md):
    
    if (i == j):
        return 0, md
    
    alp_i = md.Alpha[i]
    alp_j = md.Alpha[j]
    y_i = md.Y[i]
    y_j = md.Y[j]
    E_i = md.Error[i]
    E_j = md.Error[j]
    s = y_i*y_j
    
    if (y_i != y_j):
        L = max(0, alp_j - alp_i)
        H = min(md.C, md.C + alp_j - alp_i)
    elif (y_i == y_j):
        L = max(0, alp_i + alp_j - md.C)
        H = min(md.C, alp_i + alp_j)
      
    if (L == H):
        return 0, md
    
    k_ij = kernel(md.X[i], md.X[j])
    k_ii = kernel(md.X[i], md.X[i])
    k_jj = kernel(md.X[j], md.X[j])
    
    eta = k_ii + k_jj - 2 * k_ij

    if (eta >= 0):
        alp_j_new = alp_j + y_j * (E_i - E_j) / eta
        if (alp_j_new >= H):
            alp_j_new = H
        elif L < alp_j_new < H :
            alp_j_new = alp_j_new
        elif (alp_j_new <= L) :
            alp_j_new = L
    else:
        alp_cp = md.Alpha.copy()
        
        alp_cp[j] = L
        Lobj =  obj_func(md.X, md.Y, alp_cp) 
        
        alp_cp[j] = H
        Hobj =  obj_func(md.X, md.Y, alp_cp) 
        
        if Lobj > (Hobj + eps):
            alp_j_new = L
        elif Lobj < (Hobj - eps):
            alp_j_new = H
        else:
            alp_j_new = alp_j
            
    if (abs(alp_j_new - alp_j) < eps*(alp_j_new+alp_j+eps)):
        return 0, md

    alp_i_new = alp_i + s*(alp_j - alp_j_new)
            
    b_i = E_i + (y_i * (alp_i_new - alp_i) * k_ii) + (y_j * (alp_j_new - alp_j) * k_ij)
    b_j = E_j + (y_i * (alp_i_new - alp_i) * k_ij) + (y_j * (alp_j_new - alp_j) * k_jj)
    
    if 0 < alp_i_new and alp_i_new < md.C:
        b_new = b_i
    elif 0 < alp_j_new and alp_j_new < md.C:
        b_new = b_j
    else:
        b_new = (b_i + b_j) * 0.5
    
    md.Alpha[i] = alp_i_new
    md.Alpha[j] = alp_j_new
         
    for idx, alp in zip([i, j], [alp_i_new, alp_j_new]):
        if 0.0 < alp < md.C:
            md.Error[idx] = 0.0
    
    non_opt = [n for n in range(md.m) if (n != i and n != j)]
    md.Error[non_opt] = md.Error[non_opt] + \
                            y_i*(alp_i_new - alp_i)*kernel(md.X[i], md.X[non_opt]) + \
                            y_j*(alp_j_new - alp_j)*kernel(md.X[j], md.X[non_opt]) + md.b - b_new
    
    md.b = b_new
    
    return 1, md

def examineExample(j, md):
    
    y_j = md.Y[j]
    alp_j = md.Alpha[j]
    E_j = md.Error[j]
    r_j = E_j * y_j

    if ((r_j < -tol and alp_j < md.C) or (r_j > tol and alp_j > 0)):
        
        if len(md.Alpha[(md.Alpha != 0) & (md.Alpha != md.C)]) > 1:
            if md.Error[j] > 0:
                i = np.argmin(md.Error)
            elif md.Error[j] <= 0:
                i = np.argmax(md.Error)
            step_result, md = takeStep(i, j, md)
            if step_result:
                return 1, md
            
        for i in np.roll(np.where((md.Alpha != 0) & (md.Alpha != md.C))[0],
                          np.random.choice(np.arange(md.m))):
            step_result, md = takeStep(i, j, md)
            if step_result:
                return 1, md
        for i in np.roll(np.arange(md.m), np.random.choice(np.arange(md.m))):
            step_result, md = takeStep(i, j, md)
            if step_result:
                return 1, md
    
    return 0, md

def routine(md):
    
    numChanged = 0
    examineAll = 1

    while(numChanged > 0) or (examineAll):
        numChanged = 0
        if examineAll:
            for j in range(md.Alpha.shape[0]):
                examine_result, md = examineExample(j, md)
                numChanged += examine_result
                if examine_result:
                    obj_result = obj_func(md.X, md.Y, md.Alpha)
                    md._obj.append(obj_result)
        else:
            for j in np.where((md.Alpha != 0) & (md.Alpha != md.C))[0]:
                examine_result, md = examineExample(j, md)
                numChanged += examine_result
                if examine_result:
                    obj_result = obj_func(md.X, md.Y, md.Alpha)
                    md._obj.append(obj_result)
        if examineAll == 1:
            examineAll = 0
        elif numChanged == 0:
            examineAll = 1
        
    return md

#X_tr, Y_tr = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1)
X_tr, Y_tr = make_circles(n_samples=500, noise=0.1, factor=0.1, random_state=1)
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr, Y_tr)

Y_tr[Y_tr == 0] = -1

# Set model parameters and initial values
C = 20
m = len(X_tr_scaled)
initial_Alpha = np.zeros(m)
initial_Error =  np.zeros(m)
initial_b = 0.0

# Set tolerances
tol = 0.01 
eps = 0.01 

# Instantiate model
md = SMO_md(X_tr_scaled,
            Y_tr, 
            C, 
            initial_Alpha, 
            initial_Error,
            initial_b,
            m)

initial_Error = decision_func(md.X, md.Y, md.Alpha, md.b, md.X) - md.Y
md.Error = initial_Error


output = routine(md)
fig, ax = plt.subplots()
grid, ax = plot_decision_boundary(output, ax)
m




















