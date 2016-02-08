# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:19:27 2016

@author: asat
"""
from sklearn import preprocessing
#from sklearn.preprocessing import shuffle 
from sklearn.datasets import load_boston
from sklearn import cross_validation
boston = load_boston()
X, y = boston.data, boston.target
X_scaled = preprocessing.scale(X)
from sklearn.cross_validation import KFold
kf = KFold(506, n_folds=5, shuffle=True, random_state = 42)

import numpy as np
sizes = np.linspace(1, 10, 200)

i = 0
m1 = 0
index=0
p = 0
from sklearn.neighbors import KNeighborsRegressor

lst = [[0,0]]
while i < 200:
    ps = sizes[i]
    clr = KNeighborsRegressor(n_neighbors=5, weights='distance', p=ps)
    scores = cross_validation.cross_val_score(clr, X_scaled, y, cv=kf, scoring = 'mean_squared_error')
    m = abs(scores.mean())
    
    lst.append([i,m])  
    if m > m1: 
        m1 = m 
        index = i
        p = ps
    i = i+1

