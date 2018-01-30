#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:55:40 2018

@author: noch
"""
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from data_prepared import read_data, prepare_data, prepare_data_bi, split_data
from pegasos import pegasos_, pegasos_ker
#from perceptron import perceptron
from testing import test_with_id, test, test_ker_id, test_ker


'''

#---------
isTr = 0

b = [-0.250000625002, -0.200000666669, -0.200000400001]
num_char = [6, 6, 5]
       
f= open("/Users/noch/Documents/workspace/data_challenge/result/Yte_pegasos_6_5.csv","a+")       

for i in range (3) :
    
    Xte = read_data("Xte"+str(i), isTr)
    Xte['Id'] = pd.DataFrame({'Id':range(i*1000,(i+1) * 1000)})
    Xte_p = prepare_data(pd.DataFrame(Xte['DNA']), num_char[i])
    Xte_p['Id'] = Xte['Id']

    w = pd.read_csv("/Users/noch/Documents/workspace/data_challenge/result/w_" + str(i) + ".txt", index_col=False).as_matrix()
    

    result = test_with_id(w, b[i], Xte_p)
    
    s = ""
    
    for index, row in result.iterrows():
        
        s = s + str(int(row['Id'])) + "," + str(int(row['Bound'])) + "\n"
    
    f.write(s)
    
f.close()

#---------
#s = "Id,Bound\n"

f= open("/Users/noch/Documents/workspace/data_challenge/result/Yte_pegasos_5.csv","a+")       
nm_char = [5, 5, 5]
lmda = [10**(-5), 10**(-5), 10**(-5)]
epoch = [400000, 400000, 400000]


for i in range (3) :
    isTr = 1
    Xtr = read_data("Xtr"+str(i), isTr)
    Ytr = read_data("Ytr"+str(i), isTr)
    Ytr['Bound'][Ytr['Bound'] == 0] = -1
    
    isTr = 0
    Xte = read_data("Xte"+str(i), isTr)
    Xte['Id'] = pd.DataFrame({'Id':range(i*1000, (i+1) * 1000)})
    
    #for k in range(4):
        
    Xtr_p = prepare_data(Xtr, nm_char[i])
    Xtr_p['Bound'] = Ytr['Bound']
    
    Xte_p = prepare_data(pd.DataFrame(Xte['DNA']), nm_char[i])
    Xte_p['Id'] = Xte['Id']
    
    
    #shuffle testing set
    Xtr_p = Xtr_p.sample(frac=1)
    
    tr_X = pd.DataFrame.as_matrix(Xtr_p.iloc[:,:-1])
    tr_Y = pd.DataFrame.as_matrix(Xtr_p['Bound'])
    
    #w, b = perceptron(tr_X, tr_Y)
    w, b = pegasos_(tr_X, tr_Y, lmda[i], epoch[i])
    
    result = test_with_id(w, b, Xte_p)
    #result = result.sort_values(by=['Id']).reset_index(drop=True)
    s = ""
    for index, row in result.iterrows():
        
        s = s + str(int(row['Id'])) + "," + str(int(row['Bound'])) + "\n"
    f.write(s)
    
f.close()

'''
#---------
isTr = 1
for i in range (3) :
    
    
    X = read_data("Xtr"+str(i), isTr)
    Y = read_data("Ytr"+str(i), isTr)
    max_info = ""
    max_predic = 0
    #max_w = []
    
    Y['Bound'][Y['Bound'] == 0] = -1
     
    f= open("/Users/noch/Documents/workspace/data_challenge/result/console_pegasos_" + str(datetime.now()) + ".txt","a+")       
    print("\n testing on Xtr" +str(i)+ ", Ytr" +str(i))
    
    for k in range(4,6):
        #k = 1
    
        #data_4 = prepare_data_bi(X, k+1)
        #data_5 = prepare_data_bi(X, k+2) 
        #data_6 = prepare_data_bi(X, k+3)
        
        #concate dataframes with the same # of rows
        #data_new = pd.concat([data_4, data_5, data_6], axis=1)
        
        #data_new = prepare_data_bi(X, k+1)
        data_new = prepare_data(X, k+1)
        
        data_new['Bound'] = Y['Bound']
        
        data_train,  data_test = split_data(data_new, 70)
        
        tr_X = pd.DataFrame.as_matrix(data_train.iloc[:,:-1])
        tr_Y = pd.DataFrame.as_matrix(data_train['Bound'])
        
        te_X = pd.DataFrame.as_matrix(data_test.iloc[:,:-1])
        te_Y = pd.DataFrame.as_matrix(data_test['Bound'])
        
    #    w, b = perceptron(tr_X, tr_Y)
        
        print("number of char:" + str(k+1))
        
        for ep in range(100000,600000,100000):
            for j in range(5,7):
                
                lmd=10**(-j)
                
                #w, b = pegasos_(tr_X, tr_Y, lmd, ep) 
                alpha = pegasos_ker(tr_X, tr_Y, lmd, ep)
                
                b=0
                Y_predicted_tr = test_ker(tr_X, tr_Y, tr_X, alpha)
                Y_predicted_te = test_ker(tr_X, tr_Y, te_X, alpha)
                
                #Y_predicted_tr = test(w, b, tr_X)
                #Y_predicted_te = test(w, b, te_X)
                
                predicted_score_tr = accuracy_score(Y_predicted_tr, tr_Y, normalize=False)/len(Y_predicted_tr)
                predicted_score_te = accuracy_score(Y_predicted_te, te_Y, normalize=False)/len(Y_predicted_te)
                
                st_info = "\n test on Xtr" +str(i)+ ", Ytr" +str(i)+ "\n epoch: " + str(ep) + "\n lamda: " +str(lmd) + "\n number of character: " + str(k+1)
                
                if(predicted_score_te > max_predic):
                    max_predic = predicted_score_te
                    max_info = "\n max_result_tr: "+ str(predicted_score_tr) + st_info + "\n value of b: " + str(b) + "\n"
                    #max_w = np.asarray(w)
                
                f.write("---------------------------------------")
                f.write(st_info)
                
                f.write("\n result_tr: " 
                      + str(accuracy_score(Y_predicted_tr, tr_Y, normalize=False)) + 
                      "/" + str(len(Y_predicted_tr)) 
                      + " = " + str(predicted_score_tr))
                
                f.write("\n result_te: " 
                      + str(accuracy_score(Y_predicted_te, te_Y, normalize=False)) + 
                      "/" + str(len(Y_predicted_te))
                      + " = " + str(predicted_score_te)+"\n\n")
    f.write("****************************************************************************************************************")
    f.write("\n max_result_te: " + str(max_predic))
    f.write(max_info)
    #np.savetxt("/Users/noch/Documents/workspace/data_challenge/result/w_" + str(i) + ".txt", max_w)
        
    f.close()

#'''






