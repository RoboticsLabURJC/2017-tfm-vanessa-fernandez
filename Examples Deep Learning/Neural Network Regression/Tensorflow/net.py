"""

Neural Network Regression example
Code from tutorial: https://medium.com/@rajatgupta310198/getting-started-with-neural-network-for-regression-and-tensorflow-58ad3bd75223
Dataset from Yahoo finance: https://in.finance.yahoo.com/quote/%5EDJI/history?p=%5EDJI&guccounter=1

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Read data set using pandas
df = pd.read_csv('data.csv')

# Overview of dataset
print(df.info())

# Drop Date feature
df = df.drop(['Date'], axis=1)

# Remove all nan entries
df = df.dropna(inplace=False)

# Drop Adj close and volume feature
df = df.drop(['Adj Close','Volume'], axis=1)

# We separate the data
# 60% training data and 40% testing data
df_train = df[:int(0.6*df.shape[0])]
df_test = df[int(0.6*df.shape[0]):]
# For normalizing dataset
scaler = MinMaxScaler()

# We want to predict Close value of stock
X_train = scaler.fit_transform(df_train.drop(['Close'], axis=1).as_matrix())
y_train = scaler.fit_transform(df_train['Close'].as_matrix())
