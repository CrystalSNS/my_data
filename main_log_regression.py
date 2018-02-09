#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:20:37 2018

@author: michi
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


from data_prepared import read_data, prepare_data_no_div, prepare_data_bi, split_data
from regression import logistic_regression, sigmoid, test, predict

#---------
isTr = 1
f= open("/Users/noch/Documents/workspace/data_challenge/result/regression.txt","a+")       
#for i in range (1) :

i = 2    
    
X = read_data("Xtr"+str(i), isTr)
Y = read_data("Ytr"+str(i), isTr)
max_info = ""
max_predic = 0

Y['Bound'][Y['Bound'] == 0] = -1
 
print("\n testing on Xtr" +str(i)+ ", Ytr" +str(i))

for k in range(3,6):
    data_new = prepare_data_no_div(X, k+1)
        
    data_new['Bound'] = Y['Bound']
        
    data_train,  data_test = split_data(data_new, 70)
        
    X_tr = pd.DataFrame.as_matrix(data_train.iloc[:,:-1])
    Y_tr = pd.DataFrame.as_matrix(data_train['Bound']).astype(float).tolist()
        
    X_te = pd.DataFrame.as_matrix(data_test.iloc[:,:-1])
    Y_te = pd.DataFrame.as_matrix(data_test['Bound'])
        
    Prob_Tr = logistic_regression(X_tr, Y_tr, num_steps = 50, learning_rate = 5e-5, add_intercept=True)
    Prob_Te = logistic_regression(X_te, Y_te, num_steps = 50, learning_rate = 5e-5, add_intercept=True)                 

    X_tr = pd.DataFrame.as_matrix(data_train.iloc[:,:-1])
    X_te = pd.DataFrame.as_matrix(data_test.iloc[:,:-1])
    
    p_Tr = predict (X_tr, Prob_Tr)
    p_Te = predict (X_te, Prob_Te)
    
    Y_predicted_tr = test(p_Tr)
    Y_predicted_te = test(p_Te)
    
    predicted_score_tr = accuracy_score(Y_predicted_tr, Y_tr, normalize=False)/len(Y_predicted_tr)
    predicted_score_te = accuracy_score(Y_predicted_te, Y_te, normalize=False)/len(Y_predicted_te)
    
    st_info = "\n test on Xtr" +str(i)+ ", Ytr" +str(i) + "\n number of character: " + str(k+1)
    
    if(predicted_score_te > max_predic):
        max_predic = predicted_score_te
        max_info = "\n max_result_tr: "+ str(predicted_score_tr)  + "\n"
        
        f.write("---------------------------------------")
        f.write(st_info)
        
        f.write("\n result_tr: " 
              + str(accuracy_score(Y_predicted_tr, Y_tr, normalize=False)) + 
              "/" + str(len(Y_predicted_tr)) 
              + " = " + str(predicted_score_tr))
        
        f.write("\n result_te: " 
              + str(accuracy_score(Y_predicted_te, Y_te, normalize=False)) + 
              "/" + str(len(Y_predicted_te))
              + " = " + str(predicted_score_te)+"\n\n")
f.write("****************************************************************************************************************")
f.write("\n max_result_te: " + str(max_predic))
f.write(max_info)
   
f.close()
        
        