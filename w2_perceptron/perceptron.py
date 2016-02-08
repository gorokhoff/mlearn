# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 20:28:12 2016

@author: asat
"""

import pandas
nm  = ['t','1','2']
train = pandas.read_csv('perceptron-train.csv', names = nm)
test = pandas.read_csv('perceptron-test.csv', names = nm)
X_test = test.drop('t', axis=1)
y_test  = test[['t']]
X_train = train.copy()
X_train = X_train.drop('t', axis=1)
y_train = train[['t']]
from sklearn.linear_model import Perceptron
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
#import numpy as np
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test , pred)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf_s = Perceptron(random_state=241)
clf_s.fit(X_train_scaled, y_train)
pred_s = clf_s.predict(X_test_scaled)
score_s = accuracy_score(y_test , pred_s)
dif = score_s - score