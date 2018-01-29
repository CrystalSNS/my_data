#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:42:41 2018

@author: noch
"""
import numpy as np
import pandas as pd
from random import randint
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


def generate_points(x1,y1,x2,y2,num):
    
    df1 = pd.DataFrame(columns=['x', 'y', 'class'])
    df2 = pd.DataFrame(columns=['x', 'y', 'class'])
    for i in range(0, 0+num):
            x = randint(x1, x1+10)
            y = randint(y1, y1+10)
            df1.loc[i] =[x, y, -1]
    for i in range(21, 21+num):
            x = randint(x2, x2+10)
            y = randint(y2, y2+10)
            df2.loc[i] =[x, y, 1]
    return df1.append(df2)
            
def perceptron(X, Y):

    w = np.zeros(len(X[0]))
    eta = 0.05
    epoch = 100
    b=0
    for t in range(epoch):
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i] + b) <= 0:
                w = w + eta*X[i]*Y[i]
                b = b + eta*Y[i]
    return w,b

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

            w = np.dot((1 - (lmda*eta)),w) + np.dot((eta*Y[i][0]),X[i])
            
            b = (b*(1-(lmda*eta))) + eta*Y[i]
        
        elif ((np.dot(X[i], w) + b)*Y[i]) >= 1:    
            w = np.dot((1 - (lmda*eta)),w)
            b = b*(1-(lmda*eta))
        
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
    
def plot_points(data):
    for index, row in data.iterrows():
        # Plot the negative samples
        if row['class'] == -1:
            plt.scatter(row['x'], row['y'], s=120, marker='_', linewidths=2, color='red')
        # Plot the positive samples
        else:
            plt.scatter(row['x'], row['y'], s=120, marker='+', linewidths=2, color='blue')

def plot_points_test(data_test,Y_predicted):
    for index, row in data_test.iterrows():
        # Plot the negative samples
        if (row['class'] == -1 and row['class'] == Y_predicted[index]) :
            plt.scatter(row['x'], row['y'], s=120, marker='_', linewidths=2, color='black')
            
        elif (row['class'] == -1 and row['class'] != Y_predicted[index]) :
            plt.scatter(row['x'], row['y'], s=120, marker='_', linewidths=2, color='blue')
            
        elif (row['class'] == 1 and row['class'] == Y_predicted[index]) :
            plt.scatter(row['x'], row['y'], s=120, marker='+', linewidths=2, color='black')
            
        elif (row['class'] == 1 and row['class'] != Y_predicted[index]) :
            plt.scatter(row['x'], row['y'], s=120, marker='+', linewidths=2, color='red') 
            
            
def main():
    df = generate_points(-4, -4, 4, 4, 20)
    df = df.sample(frac=1).reset_index(drop=True)
    X = pd.DataFrame.as_matrix(df.iloc[:,0:2])
    Y = pd.DataFrame.as_matrix(df.iloc[:,2:3])
    
    plot_points(df)
    
    #w, b = perceptron(X, Y)
    w, b = pegasos_(X, Y,0.005,1000) 
    
    Y_predicted = test(w, b, X)
    
    print("\n \n Result:" + str(accuracy_score(Y_predicted, Y, normalize=False)) + 
          "/" + str(len(Y_predicted)))

    #plot_points_test(df_test, Y_predicted)
    
    #plt.plot([-1,np.dot(-1, w)],[1,np.dot(1, w)])

main()





