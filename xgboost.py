#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:02:55 2018

@author: noch
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import accuracy_score
from data_prepared import read_data, prepare_data_div, prepare_data_no_div, split_data

isTr = 1
for i in range (2,3) :
    X = read_data("Xtr"+str(i), isTr)
    Y = read_data("Ytr"+str(i), isTr)
    max_info = ""
    max_predic = 0
    
    #Y['Bound'][Y['Bound'] == 0] = -1
     
    print("\n testing on Xtr" +str(i)+ ", Ytr" +str(i))
    
    for k in range(3,4):
        print("\n number of char:"+str(k+1))
        data_new = prepare_data_div(X, k+1)
        data_new['Bound'] = Y['Bound']
        
        data_train,  data_test = split_data(data_new, 70)
        
        X_tr = data_train.iloc[:,:-1]
        Y_tr = data_train['Bound']
        
        X_te = data_test.iloc[:,:-1]
        Y_te = data_test['Bound']
        
        
        print("\n finished preparing number of char:" + str(k+1))
        dtrain = xgb.DMatrix(X_tr.values, Y_tr.values)
        dtest = xgb.DMatrix(X_te.values)
        
        print(dtest[0])
        
        # set xgboost params
        param = {
            'booster': 'gblinear',
            'silent': 1,  # logging mode - quiet
            #'objective': 'reg:linear',
            'objective': "binary:logistic", 
            'alpha': 0.0001, 
            'lambda': 1
            }  
        num_round = 2  # the number of training iterations
        
        # ------------- svm file ---------------------
        # training and testing - svm file
        bst = xgb.train(param, dtrain, num_round)
        preds = bst.predict(dtest)
        
        # extracting most confident predictions
        best_preds = [np.argmax(line) for line in preds]
        
        predicted_score_te = accuracy_score(best_preds, Y_te, normalize=False)/len(best_preds)
            
        print("\n result_te: " 
                  + str(accuracy_score(best_preds, Y_te, normalize=False)) + 
                  "/" + str(len(best_preds))
                  + " = " + str(predicted_score_te) + "\n\n")
        break         
            
            
            
            
            
















