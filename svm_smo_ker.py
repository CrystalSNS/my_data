#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:45:12 2018

@author: noch
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from data_prepared import read_data, prepare_data, split_data

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
        
def kernel__(x, y, b=1):
    #linear_kernel
    return np.dot(x, y.T) + b 
def kernel(x, y, z=1):
    #degree-2 polynomials
    return (np.dot(x, y.T) + z)**2                           
def kernel_(x, y, sigma=1):
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
    result = (Alpha * Y_tr) @ kernel(X_tr, X_te) - b
    
    return result
    
def takeStep(i, j, md):
    
    if (i == j):
        return 0, md
    
    alp_i = md.Alpha[i]
    alp_j = md.Alpha[j]
    y_i = md.Y[i]
    y_j = md.Y[j]
    E_i = md.Error[i]
    E_j = md.Error[j]
    s = y_i * y_j
    
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
    
    eta = 2 * k_ij - k_ii - k_jj 
    #print("eta: " +str(eta))
    if (eta < 0):
        alp_j_new = alp_j - y_j * (E_i - E_j) / eta
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
        #print("H: "+str(H))
        #print("L: "+str(L))
        
        
        if Lobj > (Hobj + eps):
            alp_j_new = L
        elif Lobj < (Hobj - eps):
            alp_j_new = H
        else:
            alp_j_new = alp_j
    #print("alp_j_new: "+str(alp_j_new))
    
    if alp_j_new < 1e-8:
        alp_j_new = 0.0
    elif alp_j_new > (md.C - 1e-8):
        alp_j_new = md.C
        
    if (abs(alp_j_new - alp_j) < eps * ( alp_j_new + alp_j + eps )):
        return 0, md

    alp_i_new = alp_i + s * (alp_j - alp_j_new)
            
    b_i = E_i + y_i * (alp_i_new - alp_i) * k_ii + y_j * (alp_j_new - alp_j) * k_ij + md.b
    b_j = E_j + y_i * (alp_i_new - alp_i) * k_ij + y_j * (alp_j_new - alp_j) * k_jj + md.b
    
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


def predict(md, X_te):
    
    print("b: " + str(md.b))
    for ap in md.Alpha:
        
        print("alpha: " + str(ap))
    
        
    Y_predicted = []
    for i, x_i in enumerate(X_te):
       # print("sum: " + str(np.sum((md.Alpha * md.Y) * kernel(md.X, x_i))))
        result = np.sum((md.Alpha * md.Y) * kernel(md.X, x_i)) - md.b
        #print("result: " + str(result))
        if result <=0 :
            Y_predicted.append(-1)
        elif result > 0:
            Y_predicted.append(1)
    return Y_predicted
    

isTr = 1
for i in range (3) :
    
    X = read_data("Xtr"+str(i), isTr)
    Y = read_data("Ytr"+str(i), isTr)
    
    max_info = ""
    max_predic = 0
    
    Y['Bound'][Y['Bound'] == 0] = -1
     
    f= open("/Users/noch/Documents/workspace/data_challenge/result/console_svm_SMO_ker_linear.txt","a+")       
    #f= open("/home/jibril/Desktop/data_challenge/result/console_svm_SMO_ker_linear.txt","a+")   
    
    print("\n testing on Xtr" +str(i)+ ", Ytr" +str(i))
    
    for k in range(2,3):
        
        data_new = prepare_data(X, k+1)
        
        data_new['Bound'] = Y['Bound']
        
        data_train,  data_test = split_data(data_new, 70)
        
        X_tr = np.asarray(pd.DataFrame.as_matrix(data_train.iloc[:,:-1]), dtype=float)
        Y_tr = pd.DataFrame.as_matrix(data_train['Bound'])
        
        #scaler = StandardScaler()
        #X_tr = scaler.fit_transform(X_tr, Y_tr)
        
        X_te = np.asarray(pd.DataFrame.as_matrix(data_test.iloc[:,:-1]), dtype=float)
        Y_te = pd.DataFrame.as_matrix(data_test['Bound'])
        
        #X_te = scaler.fit_transform(X_te, Y_te)
        
        m = len(X_tr)
        initial_Alpha = np.zeros(m)
        initial_Error =  np.zeros(m)
        initial_b = 0.0
        
        tol = 0.01 
        eps = 0.01 
        
        print("\n finished preparing number of char:" + str(k+1))
            
        #C_arr = [4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 1e-1, 1e-2]
        #C_arr = [10000, 1000, 100, 10, 5, 1, 0.01]
        C_arr = [0.1]
        
        for C in C_arr:
            
                # Instantiate model
                md = SMO_md(X_tr,
                            Y_tr, 
                            C, 
                            initial_Alpha, 
                            initial_Error,
                            initial_b,
                            m)            
                
                initial_Error = decision_func(md.X, md.Y, md.Alpha, md.b, md.X) - md.Y
                md.Error = initial_Error
                
                output_md = routine(md)
                
                Y_predicted_tr = predict(output_md, X_tr)
                Y_predicted_te = predict(output_md, X_te)
                
                
                Y_tr = [float(i) for i in Y_tr]
                Y_te = [float(i) for i in Y_te]
                predicted_score_tr = accuracy_score(Y_predicted_tr, Y_tr, normalize=False)/len(Y_predicted_tr)
                predicted_score_te = accuracy_score(Y_predicted_te, Y_te, normalize=False)/len(Y_predicted_te)
                
                st_info = "\n test on Xtr" + str(i) + ", Ytr" + str(i)+\
                          "\n C: " +str(C) +\
                          "\n number of character: " + str(k+1)
                 
                if(predicted_score_te > max_predic):
                    max_predic = predicted_score_te
                    max_info = "\n max_result_tr: "+ str(predicted_score_tr) + st_info  + "\n"
                
                f.write("---------------------------------------")
                f.write(st_info)
                
                f.write("\n result_tr: " 
                      + str(accuracy_score(Y_predicted_tr, Y_tr, normalize=False)) + 
                      "/" + str(len(Y_predicted_tr)) 
                      + " = " + str(predicted_score_tr))
                
                f.write("\n result_te: " 
                      + str(accuracy_score(Y_predicted_te, Y_te, normalize=False)) + 
                      "/" + str(len(Y_predicted_te))
                      + " = " + str(predicted_score_te) + "\n\n")
                
    f.write("****************************************************************************************************************")
    f.write("\n max_result_te: " + str(max_predic))
    f.write(max_info + "\n\n")
    f.close()
    break























