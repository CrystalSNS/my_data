#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:45:46 2018

@author: noch
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_prepared import read_data, prepare_data, split_data

isTr = 1
for i in range (3) :
    
    X = read_data("Xtr"+str(i), isTr)
    Y = read_data("Ytr"+str(i), isTr)
    
    max_info = ""
    max_predic = 0
    
    Y['Bound'][Y['Bound'] == 0] = -1
     
    f= open("/Users/noch/Documents/workspace/data_challenge/result/console_skl_svc.txt","a+")       
    
    print("\n testing on Xtr" +str(i)+ ", Ytr" +str(i))
    
    for k in range(2,6):
        
        data_new = prepare_data(X, k+1)
        
        data_new['Bound'] = Y['Bound']
        
        data_train,  data_test = split_data(data_new, 70)
        
        X_tr = np.asarray(pd.DataFrame.as_matrix(data_train.iloc[:,:-1]), dtype=float)
        Y_tr = pd.DataFrame.as_matrix(data_train['Bound']).astype(float).tolist()
        
        X_te = np.asarray(pd.DataFrame.as_matrix(data_test.iloc[:,:-1]), dtype=float)
        Y_te = pd.DataFrame.as_matrix(data_test['Bound'])
        
        
        print("\n finished preparing number of char:" + str(k+1))
            
        #C_arr = [10000, 1000, 100, 10, 5, 1, 0.01]
        C_arr = [1, 0.1, 0.2]
        
        for C in C_arr:
            
            SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,
                   decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                   max_iter=-1, probability=False, random_state=None, shrinking=True,
                   tol=0.001, verbose=False)
            clf = SVC()
            clf.fit(X_tr, Y_tr) 
               
            Y_predicted_tr = clf.predict(X_tr)
            Y_predicted_te = clf.predict(X_te)
            
            
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