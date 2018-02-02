#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:00:46 2018

@author: noch
"""

import pandas as pd
from sklearn.metrics import accuracy_score

from data_prepared import read_data, prepare_data, split_data
from testing import test_ker
#from svm import svm_f
#from  svm_smo_ker import smo_ker_f
from svm_ker import svm_f


isTr = 1
for i in range (3) :
    X = read_data("Xtr"+str(i), isTr)
    Y = read_data("Ytr"+str(i), isTr)
    max_info = ""
    max_predic = 0
    
    Y['Bound'][Y['Bound'] == 0] = -1
     
    f= open("/Users/noch/Documents/workspace/data_challenge/result/console_svm_ker_gaussi.txt","a+")       
    #f= open("/home/jibril/Desktop/data_challenge/result/console_svm.txt","a+")   
    print("\n testing on Xtr" +str(i)+ ", Ytr" +str(i))
    
    for k in range(2,6):
        
        data_new = prepare_data(X, k+1)
        
        data_new['Bound'] = Y['Bound']
        
        data_train,  data_test = split_data(data_new, 70)
        
        X_tr = pd.DataFrame.as_matrix(data_train.iloc[:,:-1])
        Y_tr = pd.DataFrame.as_matrix(data_train['Bound']).astype(float).tolist()
        
        X_te = pd.DataFrame.as_matrix(data_test.iloc[:,:-1])
        Y_te = pd.DataFrame.as_matrix(data_test['Bound']).astype(float).tolist()
        
        
        print("\n finished preparing number of char:" + str(k+1))
            
        C_arr = [2, 1.5, 1, 1e-1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        #C_arr = [0.01]
        
        for C in C_arr:
            
            for z in range(1, 4):
            
                alpha, b = svm_f(X_tr, Y_tr, C, z) 
                #alpha, b = smo_ker_f(X_tr, Y_tr, 1000, 1e-4, C)
        
                Y_predicted_tr = test_ker(X_tr, Y_tr, X_tr, alpha, b, z)
                Y_predicted_te = test_ker(X_tr, Y_tr, X_te, alpha, b, z)
    
                predicted_score_tr = accuracy_score(Y_predicted_tr, Y_tr, normalize=False)/len(Y_predicted_tr)
                predicted_score_te = accuracy_score(Y_predicted_te, Y_te, normalize=False)/len(Y_predicted_te)
                
                st_info = "\n test on Xtr" + str(i) + ", Ytr" + str(i)+\
                          "\n C: " +str(C) +\
                          "\n z_sigma: " + str(z) +\
                          "\n number of character: " + str(k+1)
                 
                if(predicted_score_te > max_predic):
                    max_predic = predicted_score_te
                    max_info = "\n max_result_tr: "+ str(predicted_score_tr) + st_info + "\n value of b: " + str(b) + "\n"
                
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





