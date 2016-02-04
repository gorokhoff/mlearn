# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:32:34 2016

@author: gorokhov_sa
"""

import pandas
data = pandas.read_csv('e:\\Spyder\\titanic\\titanic.csv', index_col='PassengerId')
data1 = data[['Pclass','Fare','Age','Sex','Survived']]
data2 = data1[pandas.notnull(data1['Age'])]
data2 = data2[pandas.notnull(data2['Sex'])]
data2 = data2[pandas.notnull(data2['Fare'])]
data2 = data2[pandas.notnull(data2['Pclass'])]
train = data2.copy()
train = train.drop('Survived', axis=1)
target = data2[['Survived']]
from sklearn import tree
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
train.Sex = le_sex.fit_transform(train.Sex)

clf = tree.DecisionTreeClassifier(random_state=241)
clf.fit(train,target)
imp=clf.feature_importances_
