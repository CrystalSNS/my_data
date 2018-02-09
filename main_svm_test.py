#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 20:54:18 2018

@author: noch
"""

import pandas as pd
from data_prepared import read_data, prepare_data_no_div
from testing import test_ker_id
from svm_ker import svm_ker_func




f= open("/Users/noch/Documents/workspace/data_challenge/result/Yte_svm_poly.csv","a+")       

nm_char = []
C = [5, 5, 5]
z = []

for i in range (3) :
    isTr = 1
    Xtr = read_data("Xtr"+str(i), isTr)
    Ytr = read_data("Ytr"+str(i), isTr)
    Ytr['Bound'][Ytr['Bound'] == 0] = -1
    
    isTr = 0
    Xte = read_data("Xte"+str(i), isTr)
    Xte['Id'] = pd.DataFrame({'Id':range(i*1000, (i+1) * 1000)})
    
    Xtr_p = prepare_data_no_div(Xtr, nm_char[i])
    Xtr_p['Bound'] = Ytr['Bound']
    
    Xte_p = prepare_data_no_div(pd.DataFrame(Xte['DNA']), nm_char[i])
    Xte_p['Id'] = Xte['Id']
    
    Xtr_p = Xtr_p.sample(frac=1)
    
    X_tr = pd.DataFrame.as_matrix(Xtr_p.iloc[:,:-1])
    Y_tr = pd.DataFrame.as_matrix(Xtr_p['Bound']).astype(float).tolist()
    
    alpha, b = svm_ker_func(X_tr, Y_tr, C[i], z[i])
    
    result = test_ker_id(X_tr, Y_tr, Xte_p, alpha, b, z[i])
    result['Bound'][result['Bound'] == -1] = 0
          
    s = ""
    for index, row in result.iterrows():
        s = s + str(int(row['Id'])) + "," + str(int(row['Bound'])) + "\n"
    f.write(s)
    
f.close()






