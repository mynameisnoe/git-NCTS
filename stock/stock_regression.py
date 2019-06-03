#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Apr  9 22:03:50 2019

@author: gtx1080
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("2330.TW.csv")
data = data.dropna()
data = data.iloc[:,4].values

periods = 30

train_x = data[0:(len(data) - (len(data) % periods))]
train_x = train_x.reshape(-1, 1)  


# Using stock price for the "next day" of each 30 days batch as the dependent variable.
# increment 1 to get the "next day"
train_y = data[1:(len(data) - (len(data) % periods)) + 1] 
train_y = train_y.reshape(-1, 1)


test_x = data[-(periods + 1):]
# number to 1220
test_x = test_x[:periods]
test_x = test_x.reshape(-1, 1)

test_y = data[-(periods):]
test_y = test_y.reshape(-1, 1)

from sklearn.ensemble import RandomForestRegressor
Regressor = RandomForestRegressor()
Regressor.fit(train_x, train_y)

y_pred = Regressor.predict(test_x)

plt.plot(test_y, '*', markersize=5, label = 'Real_Value')
plt.plot(y_pred, 'o', markersize=5, label = 'Predict_Value')
plt.legend()

plt.plot(test_y, label = 'Real_value')
plt.plot(y_pred, label = 'Predict_Value')
plt.legend()