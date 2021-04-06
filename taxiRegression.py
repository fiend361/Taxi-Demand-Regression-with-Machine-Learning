# comment lines that start with two hashes instead of one are code that 
# i used for debug purposses and then found hard to remove afterwards.


import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, SimpleRNN, Flatten, LSTM, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# reading the data and targets from a sample that is one cluster 
# out of the 30 clusters in the database.
data = pd.read_csv("cluster04.csv").to_numpy() 
targets = pd.read_csv("cluster04targets.csv").to_numpy() 

# # test_data = pd.read_csv("test_data.csv").to_numpy() 
# # test_targets = pd.read_csv("test_targets.csv").to_numpy() 


# calculating the means and standard deviations of the data and the 
# targets in order to normalize the input that we will feed into the 
# neural network.   
means = data.mean(axis=0)
stds = data.std(axis=0)

targets_mean = targets.mean(axis=0)
targets_std = targets.std(axis=0)

# # print(targets_mean)
# # print(targets_std)

def norm(x, n):
	return (x - means[n]) / stds[n]

def norm_targets(x):
	return (x - targets_mean[0]) / targets_std[0]

for x in range(0, 4):
	data[:, x] = norm(data[:, x], x)

targets = norm_targets(targets[:, 0])

# # print(data)
# # print(targets)


# spliting the normalized data and targets into training sets and test sets.
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)


# # print(train_data.shape)
# # print(train_targets.shape)
# # print(test_data.shape)
# # print(test_targets.shape)


# constructing the layers of the neural network.
# the dense layers's parameters and the choice of the activation function
# was a process of experimenting and hypertuning.
i = Input(shape=(4,))
x = Dense(16, activation='tanh')(i)
x = Dense(32, activation='tanh')(x)
x = Dense(16, activation='tanh')(x)
x = Dense(2)(x)

model = Model(i, x)
model.compile(loss='mse',
			  optimizer='nadam',) #adamax, nadam, sgd

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=80)

# # r = model.fit(X[:-N//2], 
# # 			  Y[:-N//2],
# # 			  epochs=80,
# # 			  validation_data=(X[-N//2:], Y[-N//2:]),)


# ploting the result of the model's training in a loss vs epoch graph.
plt.plot(r.history["val_loss"])
plt.plot(r.history["loss"])
plt.show()


# result mean square error is 0.1097


# # plt.plot(r.history["val_loss"])
# # plt.show()


# model = keras.Sequential([
# 	keras.layers.Flatten(input_shape=(28,28)),
# 	keras.layers.Dense(128, activation="relu"),
# 	keras.layers.Dense(10, activation="softmax")
# 	])

# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# model.fit(train_images, train_labels, epochs=12)

