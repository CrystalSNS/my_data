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

def read_Xte0():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/testing_set/Xte0.csv', index_col=False, ).as_matrix()

def read_Xte1():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/testing_set/Xte1.csv', index_col=False).as_matrix()

def read_Xte2():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/testing_set/Xte2.csv', index_col=False).as_matrix()

def read_Xtr0():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Xtr0.csv', index_col=False, dtype={'DNA sequence': object})

def read_Xtr1():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Xtr1.csv', index_col=False)

def read_Xtr2():
    return pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Xtr2.csv', index_col=False)

def read_Ytr0():
    data = pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Ytr0.csv', index_col=False)
    return data[:,1]

def read_Ytr1():
    data = pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Ytr1.csv', index_col=False)
    return data[:,1]

def read_Ytr2():
    data = pd.read_csv('/Users/noch/Documents/workspace/data_challenge/dataset/training_set/Ytr2.csv', index_col=False)
    return data[:,1] 



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
                
        if (index == 10):
            break     
    
    return df    
        
    #print(df)
    

X = read_Xtr0()

result = prepare_data(X,4)






