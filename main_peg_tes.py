#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 21:17:11 2018

@author: noch
"""


import pandas as pd
from data_prepared import read_data, prepare_data_div
from testing import test_with_id
from pegasos import pegasos_




f= open("/Users/noch/Documents/workspace/data_challenge/result/Yte_pegasos_13.csv","a+")       

nm_char = [6, 6, 5]
lmda = [10**(-5), 0.0001, 10**(-5)]
epoch = [400000, 300000, 300000]

for i in range (3) :
    isTr = 1
    Xtr = read_data("Xtr"+str(i), isTr)
    Ytr = read_data("Ytr"+str(i), isTr)
    Ytr['Bound'][Ytr['Bound'] == 0] = -1
    
    isTr = 0
    Xte = read_data("Xte"+str(i), isTr)
    Xte['Id'] = pd.DataFrame({'Id':range(i*1000, (i+1) * 1000)})
    print("preparing data:"+str(i))
    Xtr_p = prepare_data_div(Xtr, nm_char[i])
    Xtr_p['Bound'] = Ytr['Bound']
    
    Xte_p = prepare_data_div(pd.DataFrame(Xte['DNA']), nm_char[i])
    Xte_p['Id'] = Xte['Id']
    
    Xtr_p = Xtr_p.sample(frac=1)
    
    X_tr = pd.DataFrame.as_matrix(Xtr_p.iloc[:,:-1])
    Y_tr = pd.DataFrame.as_matrix(Xtr_p['Bound']).astype(float).tolist()
    
    print("training on data:"+str(i))
    w, b = pegasos_(X_tr, Y_tr, lmda[i], epoch[i])
    
    print("testing on data:"+str(i))
    result = test_with_id(w, b, Xte_p)
    result['Bound'][result['Bound'] == -1] = 0
          
    s = ""
    for index, row in result.iterrows():
        s = s + str(int(row['Id'])) + "," + str(int(row['Bound'])) + "\n"
    f.write(s)
    
f.close()