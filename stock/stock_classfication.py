#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:34:03 2019

@author: gtx1080
"""


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("2330.TW.csv")
data = data.dropna()
data = data.iloc[:,4].values

periods = 30

# separate the records from stock prices excluding the latest 30 days (test data)
# Uses 1 because it's only one independent feature
# created 41 batches with 30 records with 1 feature

train_x = data[0:(len(data) - (len(data) % periods))]
train_x_batches = train_x.reshape(-1, periods, 1)  


# Using stock price for the "next day" of each 30 days batch as the dependent variable.
train_y = data[1:(len(data) - (len(data) % periods)) + 1] # increment 1 to get the "next day"
train_y_batches = train_y.reshape(-1, periods, 1)



test_x = data[-(periods + 1):]
# number to 1220
test_x = test_x[:periods]
test_x = test_x.reshape(-1, periods, 1)

test_y = data[-(periods):]
test_y = test_y.reshape(-1, periods, 1)


import tensorflow as tf
tf.reset_default_graph() # memory clean

input_layer = 1 # it's just one independent variable (feature)
hidden_layer = 100
output_layer = 1 # it's just one dependent variable

xph = tf.placeholder(tf.float32, [None, periods, input_layer])
yph = tf.placeholder(tf.float32, [None, periods, output_layer])

cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_layer, activation = tf.nn.relu)

cell_output = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=1) # Dense Neural Network

rnn_output, _ = tf.nn.dynamic_rnn(cell_output, xph, dtype=tf.float32)

error = tf.losses.mean_squared_error(labels=yph, predictions=rnn_output)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(error)


with tf.Session() as s:
    
    s.run(tf.global_variables_initializer())
    
    for epoch in range(1000):
        _, cost = s.run([train, error], feed_dict = { xph: train_x_batches, yph: train_y_batches })
        if epoch % 100 == 0:
            print('Epoch: ', epoch + 1, ' - Cost error: ', cost)
            
    predictions = s.run(rnn_output, feed_dict = { xph: test_x })


import numpy as np
check_y = np.ravel(test_y) # reduction (1,30,1) to (30,)
check_predictions = np.ravel(predictions)


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(check_y, check_predictions)

plt.plot(check_y, '*', markersize=5, label = 'Real_Value')
plt.plot(check_predictions, 'o', markersize=5, label = 'Predict_Value')
plt.legend()

plt.plot(check_y, label = 'Real_value')
plt.plot(check_predictions, label = 'Predict_Value')
plt.legend()