#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:52:19 2018

@author: noch
"""

from robsvm import robsvm
from cvxopt import matrix, normal
import pandas as pd
from sklearn.metrics import accuracy_score
from testing import test
from data_prepared import read_data, prepare_data_div, split_data

isTr = 1
for i in range (2,3) :
    X = read_data("Xtr"+str(i), isTr)
    Y = read_data("Ytr"+str(i), isTr)
    max_info = ""
    max_predic = 0
    
    Y['Bound'][Y['Bound'] == 0] = -1
     
    #f= open("/Users/noch/Documents/workspace/data_challenge/result/console_svm_ker_gaussi_C_big.txt","a+")       
    #f= open("/home/jibril/Desktop/data_challenge/result/console_svm_ker_gaussi.txt","a+")   
    print("\n testing on Xtr" +str(i)+ ", Ytr" +str(i))
    
    for k in range(2,5):
        print("\n number of char:"+str(k+1))
        data_new = prepare_data_div(X, k+1)
        
        data_new['Bound'] = Y['Bound']
        
        data_train,  data_test = split_data(data_new, 70)
        
        X_train = data_train.iloc[:,:-1]
        Y_tr = pd.DataFrame.as_matrix(data_train['Bound']).astype(float).tolist()
        
        X_te = pd.DataFrame.as_matrix(data_test.iloc[:,:-1])
        Y_te = pd.DataFrame.as_matrix(data_test['Bound']).astype(float).tolist()
        
        
        print("\n finished preparing number of char:" + str(k+1))
            
        gamma_arr = [100, 20, 10, 1, 0.1, 0.01]
        #C_arr = [0.01]
        
        
        X_tr = matrix(X_train.values.T.tolist())
        #m = X_tr.shape[0]
        #n = X_tr.shape[1]
        m,n = X_tr.size
        for gamma in gamma_arr:
            
            # generate uncertainty ellipsoids
            k = 2
            P = [0.1*normal(10*n,n) for i in range(k)]
            P = [ p.T*p for p in P]
            e = matrix(0,(m,1))
            for i in range(m):
                if Y_tr[i] == -1: e[i] = 1
            
            # solve SVM training problem
            w, b, u, v, iterations = robsvm(X_tr, Y_tr, gamma, P, e)
            #print(w)
            print("b:"+str(b))
            X_train_m = pd.DataFrame.as_matrix(X_train)
            Y_predicted_tr = test(w, b, X_train_m)
            Y_predicted_te = test(w, b, X_te)

            predicted_score_tr = accuracy_score(Y_predicted_tr, Y_tr, normalize=False)/len(Y_predicted_tr)
            predicted_score_te = accuracy_score(Y_predicted_te, Y_te, normalize=False)/len(Y_predicted_te)
            
             
            print("\n gamma:" + str(gamma))
            print("\n result_tr: " 
                  + str(accuracy_score(Y_predicted_tr, Y_tr, normalize=False)) + 
                  "/" + str(len(Y_predicted_tr)) 
                  + " = " + str(predicted_score_tr))
            
            print("\n result_te: " 
                  + str(accuracy_score(Y_predicted_te, Y_te, normalize=False)) + 
                  "/" + str(len(Y_predicted_te))
                  + " = " + str(predicted_score_te) + "\n\n")
            
    break


