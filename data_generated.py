#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:42:41 2018

@author: noch
"""
#import numpy as np
import pandas as pd
from random import randint
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from testing import test, test_ker
from pegasos import pegasos_ker, pegasos_
#from perceptron import perceptron


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
            
            
df = generate_points(10, 10, 40, 40, 20)
df = df.sample(frac=1).reset_index(drop=True)
X = pd.DataFrame.as_matrix(df.iloc[:,0:2])
#Y = pd.DataFrame.as_matrix(df.iloc[:,2:3])
Y = pd.DataFrame.as_matrix(df['class'])

plot_points(df)

#w, b = perceptron(X, Y)
#w, b = pegasos_(X, Y,0.005,1000) 
alpha = pegasos_ker(X, Y,0.005,1000) 

#Y_predicted = test(w, b, X)
Y_predicted = test_ker(X, Y, X, alpha)

print("\n \n Result:" + str(accuracy_score(Y_predicted, Y, normalize=False)) + 
      "/" + str(len(Y_predicted)))

#plot_points_test(df_test, Y_predicted)

#plt.plot([-1,np.dot(-1, w)],[1,np.dot(1, w)])






