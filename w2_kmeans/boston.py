# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:19:27 2016

@author: asat
"""
from sklearn import preprocessing
#from sklearn.preprocessing import shuffle 
from sklearn.utils import shuffle
from sklearn.datasets import load_boston

def _boston_subset(n_samples=200):
    global BOSTON
    if BOSTON is None:
        boston = load_boston()
        X, y = boston.data, boston.target
        X, y = shuffle(X, y, random_state=0)
        #X, y = X[:n_samples], y[:n_samples]
        X = preprocessing.scale(X)
        BOSTON = X, y
    return BOSTON
import numpy as np
sizes = np.linspace(1, 10, 200)

i = 0
from sklearn.neighbors import KNeighborsRegressor
while i < 200:
    neigh = KNeighborsRegressor(n_neighbors=5, n)
    
