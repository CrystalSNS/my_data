#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 22:03:44 2018

@author: noch
"""

import pandas as pd
from data_prepared import read_data, prepare_data_div
from regression import logistic_regression, test_with_id, predict
from sklearn.linear_model import LogisticRegression

i = 0 # for the 1st dataset
f= open("/Users/noch/Documents/workspace/data_challenge/result/Y/Yte_skl_lg_"+str(i)+".csv","a+")      

#i = 1 # for the 2nd dataset
#f = open("/home/jibril/Desktop/data_challenge/result/Yte_lg_"+str(i)+".csv","a+")  
 
isTr = 1
print("\n testing on Xtr" +str(i)+ ", Ytr" +str(i))
Xtr = read_data("Xtr"+str(i), isTr)
Ytr = read_data("Ytr"+str(i), isTr)

isTr = 0
Xte = read_data("Xte"+str(i), isTr)
Xte['Id'] = pd.DataFrame({'Id':range(i*1000, (i+1) * 1000)})

nm_char = 6

print("preparing data..")
Xtr_p = prepare_data_div(Xtr, nm_char)
Xtr_p['Bound'] = Ytr['Bound']

Xte_p = prepare_data_div(pd.DataFrame(Xte['DNA']), nm_char)


X_tr = pd.DataFrame.as_matrix(Xtr_p.iloc[:,:-1])
Y_tr = pd.DataFrame.as_matrix(Xtr_p['Bound']).astype(float).tolist()


print("training logistic regression..")

LogisticRegression(fit_intercept=True, C = 1e15)
clf = LogisticRegression()
clf.fit(X_tr, Y_tr) 

print("predicting the test set..")
    
result = clf.predict(X_tr)

      
s = ""
for i, row in result.iterrows():
    s = s + str(int(Xte['Id'][i])) + "," + str(int(row)) + "\n"
f.write(s)

print("finish!")

f.close()