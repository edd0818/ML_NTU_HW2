# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

X_train_dataset = pd.read_csv('data/X_train')
Y_train_dataset = pd.read_csv('data/Y_train')

# X_train = X_train_dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 3].values
X_train = X_train_dataset.iloc[:, 1:].values.astype(np.float64)
Y_train = Y_train_dataset.iloc[:].values


# Feature scaling
def normalize(dataset):
    for col in range(dataset.shape[1]):
        A = (dataset[:, col] - dataset[:, col].min())
        B = (dataset[:, col].max() - dataset[:, col].min())
        dataset[:, col] = A / B


normalize(X_train)

# Implement Logistic Regression
# Constant term as weight
dim = X_train.shape[1] + 1
w = np.ones([dim, 1])
X_train = np.concatenate((np.ones([X_train.shape[0], 1]), X_train), axis=1).astype(float)
# Training configuration
learning_rate = 100
iter_time = 1
adagrad = np.zeros([dim, 1])
eps = 0.0000000001
for t in range(iter_time):
    # loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    p = 1 / (1 + np.exp(np.dot(X_train, w)))
    print(p)
    # loss = - np.sum(np.dot(Y_train[:1], np.log(p)) + np.dot(1 - Y_train[:1], np.log(1 - p)))  # cross entropy
    # if(t%100==0):
    #     print(str(t) + ":" + str(loss))
    # gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    # adagrad += gradient ** 2
    # w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
