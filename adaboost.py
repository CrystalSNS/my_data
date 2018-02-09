#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:42:22 2018

@author: noch
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data_prepared import read_data, prepare_data_div, prepare_data_no_div, split_data
from pegasos import pegasos_
from svm_ker import svm_ker_func

def predict_pegasos(w, b, X_test):
    
    Y_predicted = []
    
    for i, x in enumerate(X_test):
        l = np.dot(X_test[i],w) + b
        if l <=0 :
            Y_predicted.append(-1.0)
        else:
            Y_predicted.append(1.0)
    return Y_predicted

def kernel(x, y, z):
    #degree-2 polynomials
    return (np.dot(x, y) + z)**2

def predict_svm_ker(X_tr, Y_tr, X_te, alpha, b, z):
    
    X_train = X_tr
    X_test = X_te
    Y_predicted = []
    for i, x_i in enumerate(X_test):
        result = np.sum(alpha * Y_tr * kernel(X_train, x_i, z)) - b
        if result <=0 :
            Y_predicted.append(-1.0)
        else:
            Y_predicted.append(1.0)
    return Y_predicted

def final_predict(Y_all, Alpha):
    
    Y_predicted = []
    for i in range(len(Y_all[0])):
        sm = 0
        for j, Y in enumerate(Y_all):
            sm = sm + Alpha[j]*Y[i]
            
        if sm <=0:
            Y_predicted.append(-1.0)
        else:
            Y_predicted.append(1.0)
            
    return Y_predicted   

#def adaboost_func():
    
isTr = 1




#for i in range (3) :

i = 2

X = read_data("Xtr"+str(i), isTr)
Y = read_data("Ytr"+str(i), isTr)

Y['Bound'][Y['Bound'] == 0] = -1
 
print("\n testing on Xtr" +str(i)+ ", Ytr" +str(i))



X['Bound'] = Y['Bound']

data_train,  data_test = split_data(X, 70)

X_train = data_train.iloc[:,:-1]
Y_tr = pd.DataFrame.as_matrix(data_train['Bound']).astype(float).tolist()

X_test = data_test.iloc[:,:-1]
Y_te = pd.DataFrame.as_matrix(data_test['Bound']).astype(float).tolist()


#num_char = 5

Alpha = []
Y_all_te = []
Y_all_tr = []

D = 0
epoch = [500000, 400000, 400000, 300000, 200000, 500000]
lmda = [1e-05, 1e-05, 0.0001, 0.0001, 0.0001, 1e-05 ]
num_char = [5, 5, 5, 5, 5, 6]

for t in range(5):
    
    epsl = 0
    Z = 0

    print("... is training on classifier pegasos"+str(t))
    X_tr = prepare_data_div(X_train, num_char[t])
    X_tr = pd.DataFrame.as_matrix(X_tr)
    D = 1/len(X_tr)
    
    X_te = prepare_data_div(X_test, num_char[t])
    X_te = pd.DataFrame.as_matrix(X_te)
    
    #based-classifier
    w, b = pegasos_(X_tr, Y_tr, lmda[t], epoch[t])
    
    Y_pre_tr = predict_pegasos(w, b, X_tr)
    Y_pre_te = predict_pegasos(w, b, X_te)
    
    predicted_sco_tr = accuracy_score(Y_pre_tr, Y_tr, normalize=False)/len(Y_pre_tr)
    print("predicted_score_tr:"+str(predicted_sco_tr))
    
    predicted_sco_te = accuracy_score(Y_pre_te, Y_te, normalize=False)/len(Y_pre_te)
    print("predicted_score_te:"+str(predicted_sco_te))
    
    '''
    elif (t == 1):
        C = 0.1
        z = 1
        print("... is training on classifier svm - degree-2 polynomials kernel")
        X_tr = prepare_data_no_div(X_train, numchar)
        X_tr = pd.DataFrame.as_matrix(X_tr)
        
        X_te = prepare_data_no_div(X_test, numchar)
        X_te = pd.DataFrame.as_matrix(X_te)
        #based-classifier
        alpha, b = svm_ker_func(X_tr, Y_tr, C, z)
        
        Y_pre_tr = predict_svm_ker(X_tr, Y_tr, X_tr, alpha, b, z)
        Y_pre_te = predict_svm_ker(X_tr, Y_tr, X_te, alpha, b, z)
        
        predicted_sco_tr = accuracy_score(Y_pre_tr, Y_tr, normalize=False)/len(Y_pre_tr)
        print("predicted_score_tr:"+str(predicted_sco_tr))
        
        predicted_sco_te = accuracy_score(Y_pre_te, Y_te, normalize=False)/len(Y_pre_te)
        print("predicted_score_te:"+str(predicted_sco_te))
    '''    
    # Get the error epsilon over the miss-classified 
    num_err = 0
    for idx, y in enumerate(Y_tr):
        if (y != Y_pre_tr[idx]):
            epsl = epsl + D
            num_err = num_err + 1
    print("num_err:"+str(num_err)+"\n")
    if(num_err != 0):
        
        Y_all_tr.append(Y_pre_tr)
        Y_all_te.append(Y_pre_te)
        
        alp = 0.5 * math.log((1-epsl)/epsl, 2)
        
        Alpha.append(alp)
        
        for idx, y in enumerate(Y_tr):
            Z = Z + (D * np.exp(-alp * y * Y_pre_tr[idx]))
        
        for i, y in enumerate(Y_tr):
            
            D = (D * np.exp(-alp * y * Y_pre_tr[i]))/Z
                
                
if(len(Alpha) != 0):
    print("\n\n Adaboost predict ")
    Y_predicted_tr = final_predict(Y_all_tr, Alpha)
    predicted_score_tr = accuracy_score(Y_predicted_tr, Y_tr, normalize=False)/len(Y_predicted_tr)
    print("predicted_score_tr:"+str(predicted_score_tr))
    
    Y_predicted_te = final_predict(Y_all_te, Alpha)
    predicted_score_te = accuracy_score(Y_predicted_te, Y_te, normalize=False)/len(Y_predicted_te)
    print("predicted_score_te:"+str(predicted_score_te))
else:
    print("No error found!")


















