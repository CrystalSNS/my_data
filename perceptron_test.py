#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:55:40 2018

@author: noch
"""

import numpy as np
import pandas as pd
from random import randint
from itertools import product
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


def read_data(st, isTr):
    sbroot = "testing_set/"
    if(isTr): 
        sbroot = "training_set/"
    root = "/Users/noch/Documents/workspace/data_challenge/dataset/" + sbroot
    data = 0
    st = st+".csv"
    data =  pd.read_csv(root+st, index_col=False)
    
    return data

def creat_col_name(r):
    permu = product("ATGC", repeat = r)
    arr = []
    for val in permu:
        st = ""
        for j in range(r):
            st = st + val[j]
        arr.append(st)
    return arr

#prepare data one char
def prepare_data(X, num_char):

    df = pd.DataFrame(columns = creat_col_name(num_char))
    col_name = list(df)
    for index, row in X.iterrows():
        ln = len(row['DNA'])
        df.loc[index] = 0
        for i in range(ln-num_char+1):
            s = str(row['DNA'])
            for n in col_name:
                st = ""
                for t in range(num_char):
                    st = st + s[i+t]
                    
                if(st == n):
                    df.loc[index][n] = df.loc[index][n]+1
                    break
                        
        for n in col_name:
            df.loc[index][n] = df.loc[index][n]/(ln-num_char+1)
        
        #if (index == 2):
        #  break     
    
    return df    
        
    #print(df)

def split_data(df, tr_num):
    msk = np.random.rand(len(df)) < (tr_num/100)
    return (df[msk], df[~msk])


def perceptron(X, Y):

    w = np.zeros(len(X[0]))
    eta = 0.05
    epoch = 1000
    b=0
    for t in range(epoch):
        for i, x in enumerate(X):
            if ((np.dot(X[i], w)+b)*Y[i]) <= 0:
                w = w + eta*X[i]*Y[i]
                b = b + eta*Y[i]
    return w, b

def pegasos(X, Y):
    
    eta = 0
    lmda = 0.0005
    epoch = 1000
    w = np.zeros(len(X[0]))
    S_p =  []
    Y_p =  []
    b = 0

    for t in range(1,epoch):
        
        for i, x in enumerate(X):
            if ((np.dot(X[i], w) + b)*Y[i]) < 1:
                S_p.append(X[i])
                Y_p.append(Y[i])
                
        eta = 1/(lmda * t)
        sm = 0
        sm_y = 0
        for idx, row in enumerate(S_p):
            sm = sm + (S_p[idx]*Y_p[idx])
            sm_y = sm_y + Y_p[idx]
        
        w = np.dot((1 - (lmda*eta)),w) + ((eta/X.shape[0]) * sm)
        b = (b*(1-(lmda*eta))) + ((eta/X.shape[0])*sm_y)
        
    return w, b 


def pegasos_(X, Y, lmda, epoch):
    
    eta = 0
    #lmda = 0.0005
    #epoch = 1000
    w = np.zeros(len(X[0]))
    b = 0

    for t in range(1,epoch):
        eta = 1/(lmda * t)
        
        i = randint(0, X.shape[0]-1)
        
        
        if ((np.dot(X[i], w) + b)*Y[i]) < 1:

            w = np.dot((1 - (lmda*eta)),w) + np.dot((eta*Y[i]),X[i])
            
            b = (b*(1-(lmda*eta))) + eta*Y[i]
        
        elif ((np.dot(X[i], w) + b)*Y[i]) >= 1:    
            w = np.dot((1 - (lmda*eta)),w)
            b = b*(1-(lmda*eta))
        
        if(np.linalg.norm(w) != 0):
            w = min(1,( 1/np.sqrt(lmda) )/ np.linalg.norm(w) ) * w
        
    return w, b 

  
def test(w, b, X_test):
    Y_predicted = []
    for i, x in enumerate(X_test):
        if(np.dot(X_test[i],w) + b <=0 ):
            Y_predicted.append(-1)
        else:
            Y_predicted.append(1)
    return Y_predicted

def test_with_id(w, b, X):
    result = pd.DataFrame(columns = ['Id','Bound'])
    X_test = pd.DataFrame.as_matrix(X.loc[:, X.columns != 'Id'])
    Y_test = pd.DataFrame.as_matrix(X.loc[:,'Id'])
    for i, x in enumerate(X_test):
        result.loc[i] = 0
        if(np.dot(X_test[i],w) + b <=0 ):
            result.loc[i]['Id'] = Y_test[i]
            result.loc[i]['Bound'] = -1
        else:
            result.loc[i]['Id'] = Y_test[i]
            result.loc[i]['Bound'] = 1

    return result

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
     
     
    print("\n testing on Xtr" +str(i)+ ", Ytr" +str(i))
    
    f= open("/Users/noch/Documents/workspace/data_challenge/result/console_pegasos.txt","a+")       
#    for k in range(3,6):
    k = 3
    
    data_4 = prepare_data(X, k+1)
    data_5 = prepare_data(X, k+2) 
    data_6 = prepare_data(X, k+3)
    
    #concate dataframes with the same # of rows
    data_new = pd.concat([data_4, data_5, data_6], axis=1)
    
    data_new['Bound'] = Y['Bound']
    
    data_train,  data_test = split_data(data_new, 70)
    
    tr_X = pd.DataFrame.as_matrix(data_train.iloc[:,:-1])
    tr_Y = pd.DataFrame.as_matrix(data_train['Bound'])
    
    te_X = pd.DataFrame.as_matrix(data_test.iloc[:,:-1])
    te_Y = pd.DataFrame.as_matrix(data_test['Bound'])
    
#    w, b = perceptron(tr_X, tr_Y)
    
    for ep in range(100000,600000,100000):
        for j in range(4,8):
            
            lmd=10**(-j)
            
            w, b = pegasos_(tr_X, tr_Y, lmd, ep)
            
            Y_predicted_tr = test(w, b, tr_X)
            
            Y_predicted_te = test(w, b, te_X)
            
            
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






