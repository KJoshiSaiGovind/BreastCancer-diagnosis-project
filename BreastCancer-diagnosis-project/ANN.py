# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:25:41 2020

@author: Sunil Kiran
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('data.csv')
X=dataset.iloc[:,2:32].values
y=dataset.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lr1=LabelEncoder()
y=lr1.fit_transform(y)

""""ohe=OneHotEncoder(categorical_features=[1])
X=ohe.fit_transform(X).toarray()
X=X[:,1:]""""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(output_dim=16,init='uniform',activation='relu',input_dim=30))


classifier.add(Dense(output_dim=16,init='uniform',activation='relu'))


classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


classifier.fit(X_train,y_train,batch_size=10,nb_epoch=50)


y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)








