#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:28:53 2018

@author: noch
"""
import numpy as np
import random

def linear_kernel(x, y, b=1):
    return x @ y.T + b # Note the @ operator for matrix multiplication
"""Returns the linear combination of arrays `x` and `y` with
the optional bias term `b` (set to 1 by default)."""
                           


def gaussian_kernel(x, y, sigma=1):

    if np.ndim(x) == 1 and np.ndim(y) == 1:
        result = np.exp(- np.linalg.norm(x - y) / (2 * sigma ** 2))
    elif (np.ndim(x) > 1 and np.ndim(y) == 1) or (np.ndim(x) == 1 and np.ndim(y) > 1):
        result = np.exp(- np.linalg.norm(x - y, axis=1) / (2 * sigma ** 2))
    elif np.ndim(x) > 1 and np.ndim(y) > 1:
        result = np.exp(- np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], axis=2) / (2 * sigma ** 2))
    return result
"""Returns the gaussian similarity of arrays `x` and `y` with
kernel width parameter `sigma` (set to 1 by default)."""
                               

def smo_f(X, Y, mx_pas, tol, C):
    
    alpha = np.zeros(X.shape[0])
    b = 0
    
    for pas in range (mx_pas):
        
        n_changed_alpha = 0
        
        for i, x in enumerate(X):
            #Calculate E_i
            E_i = np.sum((alpha * Y * np.dot(X, X[i])) + b - Y[i])
            
            if ((Y[i]*E_i < -tol and alpha[i] < C) or (Y[i]*E_i > tol and alpha[i] > 0)):
                
                #choose i randomly where j != i
                r = []
                for t in range(0, i):
                    r.append(t)
                for t in range(i+1, X.shape[0]-1):
                    r.append(t)
                j = random.choice(r)
                
                #Calculate E_j
                E_j = np.sum((alpha * Y * np.dot(X, X[j])) + b - Y[j])
                
                #Save old alphas
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                # Compute L & H, the bounds on new possible alpha values
                if (Y[i] != Y[j]):
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                elif (Y[i] == Y[j]):
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                  
                if (L == H):
                    continue 
                

                # Compute kernel & 2nd derivative eta
                k12 = np.dot(X[i], X[j])
                k11 = np.dot(X[i], X[i])
                k22 = np.dot(X[j], X[j])
                
                eta = 2 * k12 - k11 - k22

                if (eta >= 0):
                    continue
                # Compute and clip new value alpha_j (a2) if eta is negative
                elif (eta < 0):
                    alpha_j_new = alpha_j_old - Y[j] * (E_i - E_j) / eta
                    # Clip a2 based on bounds L & H
                    if (alpha_j_new >= H):
                        alpha_j_new = H
                    elif L < alpha_j_new < H :
                        alpha_j_new = alpha_j_new
                    elif (alpha_j_new <= L) :
                        alpha_j_new = L
                        
                if (abs(alpha_j_new - alpha_j_old) < 10**-5):
                    continue
                #Determine value for alpha_i
                alpha_i_new = alpha_i_old + Y[i]*Y[j]*(alpha_j_old - alpha_j_new)
                        
                # Compute b1 and b2 
                b_1 = b + E_i + (Y[i] * (alpha_i_new - alpha_i_old) * k11) + (Y[j] * (alpha_j_new - alpha_j_old) * k12)
                b_2 = b + E_j + (Y[i] * (alpha_i_new - alpha_i_old) * k12) + (Y[j] * (alpha_j_new - alpha_j_old) * k22)
                
                #Compute b
                if 0 < alpha_i_new and alpha_i_new < C:
                    b = b_1
                elif 0 < alpha_j_new and alpha_j_new < C:
                    b = b_2
                # Average thresholds if both are bound
                else:
                    b = (b_1 + b_2) * 0.5
                
                # Update new alphas
                alpha[i] = alpha_i_new
                alpha[j] = alpha_j_new
                     
                n_changed_alpha = n_changed_alpha + 1
            
            if (n_changed_alpha == 0):
               pas = pas + 1
            else:
                pas = 0
                
        return alpha, b
    