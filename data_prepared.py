#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:04:04 2018

@author: noch
"""
import numpy as np
import pandas as pd
from itertools import product

def read_data(st, isTr):
    sbroot = "testing_set/"
    if(isTr): 
        sbroot = "training_set/"
    root = "/Users/noch/Documents/workspace/data_challenge/dataset/" + sbroot
    #root = "/home/jibril/Desktop/data_challenge/dataset/" + sbroot
    data = 0
    st = st+".csv"
    data =  pd.read_csv(root+st, index_col=False)
    
    return data

def creat_col_name(r):
    permu = product("<ATGC>", repeat = r)
    arr = []
    for val in permu:
        st = ""
        for j in range(r):
            st = st + val[j]
        arr.append(st)
    return arr

#count the # of char occured in the sequence and devided by len 
def prepare_data_div(X, num_char):

    df = pd.DataFrame(columns = creat_col_name(num_char))
    col_name = list(df)
    for index, row in X.iterrows():
        ln = len(row['DNA'])
        df.loc[index] = 0
        for i in range(ln-num_char+1):
            s = str(row['DNA'])
            s =  "<" + str(s) + ">"
            for n in col_name:
                st = ""
                for t in range(num_char):
                    st = st + s[i+t]
                    
                if(st == n):
                    df.loc[index][n] = df.loc[index][n]+1
                    break
                        
        for n in col_name:
           df.loc[index][n] = df.loc[index][n]/(ln-num_char+1)
        
        #if (index == 499):
        #   break     
    
    return df 

#count the # of char occured in the sequence 
def prepare_data_no_div(X, num_char):

    df = pd.DataFrame(columns = creat_col_name(num_char))
    col_name = list(df)
    for index, row in X.iterrows():
        ln = len(row['DNA'])
        df.loc[index] = 0
        for i in range(ln-num_char+1):
            s = str(row['DNA'])
            s =  "<" + str(s) + ">"
            for n in col_name:
                st = ""
                for t in range(num_char):
                    st = st + s[i+t]
                    
                if(st == n):
                    df.loc[index][n] = df.loc[index][n]+1
                    break
        
        #if (index == 499):
        #   break     
    
    return df    

#if char occured in the sequence it'll be 1 else 0
def prepare_data_bi(X, num_char):

    df = pd.DataFrame(columns = creat_col_name(num_char))
    col_name = list(df)
    for index, row in X.iterrows():
        #print(str(row['DNA']))
        
        ln = len(row['DNA'])
        df.loc[index] = 0
        for i in range(ln-num_char+1):
            s = str(row['DNA'])
            for n in col_name:
                st = ""
                for t in range(num_char):
                    st = st + s[i+t]
                    
                if(st == n and df.loc[index][n] == 0):
                    df.loc[index][n] = 1
                    break
                        
        #if (index == 20):
        # break     
    
    return df    
        
    #print(df)

def split_data(df, tr_num):
    msk = np.random.rand(len(df)) < (tr_num/100)
    return (df[msk], df[~msk])