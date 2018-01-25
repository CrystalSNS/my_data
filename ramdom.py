#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:18:43 2018

@author: noch
"""
from random import randint

s = "Id,Bound\n"
for i in range(0, 3000):
    s =  s + str(i) + "," + str(randint(0, 1)) + "\n"

f= open("/Users/noch/Documents/workspace/data_challenge/result/Yte_ramdomly.csv","w+")       
f.write(s)
f.close()