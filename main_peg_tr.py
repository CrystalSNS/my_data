#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:03:35 2018

@author: noch
"""

import pandas as pd
from sklearn.metrics import accuracy_score

from data_prepared import read_data, prepare_data_div, split_data
from pegasos import pegasos_
from testing import test

isTr = 1
for i in range (3) :
    
    X = read_data("Xtr"+str(i), isTr)
    Y = read_data("Ytr"+str(i), isTr)
    max_info = ""
    max_predic = 0
    
    Y['Bound'][Y['Bound'] == 0] = -1
     
    f= open("/Users/noch/Documents/workspace/data_challenge/result/console_pegasos_12_div.txt","a+")       
    #f= open("/home/jibril/Desktop/data_challenge/result/console_pegasos_12_div.txt","a+")   
    print("\n testing on Xtr" +str(i)+ ", Ytr" +str(i))
    
    for k in range(4,6):

        data_new = prepare_data_div(X, k+1)
        
        data_new['Bound'] = Y['Bound']
        
        data_train,  data_test = split_data(data_new, 70)
        
        X_tr = pd.DataFrame.as_matrix(data_train.iloc[:,:-1])
        Y_tr = pd.DataFrame.as_matrix(data_train['Bound']).astype(float).tolist()
        
        X_te = pd.DataFrame.as_matrix(data_test.iloc[:,:-1])
        Y_te = pd.DataFrame.as_matrix(data_test['Bound'])
        
        print("number of char:" + str(k+1))
        
        for ep in range(200000,600000,100000):
            for j in range(5,7):
                lmd = 10**(-j)
                
                w, b = pegasos_(X_tr, Y_tr, lmd, ep) 
                
                Y_predicted_tr = test(w, b, X_tr)
                Y_predicted_te = test(w, b, X_te)
                
                predicted_score_tr = accuracy_score(Y_predicted_tr, Y_tr, normalize=False)/len(Y_predicted_tr)
                predicted_score_te = accuracy_score(Y_predicted_te, Y_te, normalize=False)/len(Y_predicted_te)
                
                st_info = "\n test on Xtr" +str(i)+ ", Ytr" +str(i)+ "\n epoch: " + str(ep) + "\n lamda: " +str(lmd) + "\n number of character: " + str(k+1)
                 
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
                      + " = " + str(predicted_score_te)+"\n\n")
    f.write("****************************************************************************************************************")
    f.write("\n max_result_te: " + str(max_predic))
    f.write(max_info)
            
    f.close()




