# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:56:51 2016

@author: gorokhov_sa
"""
import pandas
nm = ['target','Alcohol','MalicAcid','Ash','AshAlcalinity ','Magnesium','TotalPhenols','Flavanoids','NonflavanoidPhenols','Proanthocyanins','ColorIntensity','Hue','DilutedWines','Proline']
data = pandas.read_csv('wine.data',names = nm)
train = data.copy()
train = train.drop('target', axis=1)
target = data[['target']]

#1
from sklearn import cross_validation
from numpy import mean
from sklearn.cross_validation import KFold
kf = KFold(178, n_folds=5, shuffle=True, random_state = 42)
#2
lst = [[0,0]]
from sklearn.neighbors import KNeighborsClassifier
i = 1
m1 = 0
index=0
while i <= 50:
    neigh = KNeighborsClassifier(n_neighbors=i)
    scores = cross_validation.cross_val_score(neigh, train, target, cv=kf)
    m = mean(scores)
    lst.append([i,m])    
    print(i)
    if m > m1: 
        m1 = m 
        index = i
    i = i+1
        
    
