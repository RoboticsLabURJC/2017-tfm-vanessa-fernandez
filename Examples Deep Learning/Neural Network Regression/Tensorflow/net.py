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


def neural_net_model(X_data, input_dim):
	"""
	This function applys 2 hidden layer feed forward neural net.

	Weights and biases are abberviated as W_1,W_2 and b_1, b_2 
	These are variables with will be updated during training.

	"""

	number_of_nodes_in_layer = 10

	# tf.Variable will create a variable of which value will be changing during optimization steps
	# tf.random_uniform will generate random number of uniform distribution of dimension specified ([input_dim,number_of_nodes_in_layer])
	W_1 = tf.Variable(tf.random_uniform([input_dim, number_of_nodes_in_layer]))
	# tf.zeros will create zeros of dimension specified (vector of (1,number_of_hidden_node))
	b_1 = tf.Variable(tf.zeros([number_of_nodes_in_layer]))
	# tf.add() will add two parameters
	# tf.matmul() will multiply two matrices (Weight matrix and input data matrix)
	layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
	# tf.nn.relu() is an activation function  that after multiplication and addition of weights and biases we apply activation function
	layer_1 = tf.nn.relu(layer_1)

	# layer 1 multiplying and adding bias then activation function
	W_2 = tf.Variable(tf.random_uniform([number_of_nodes_in_layer, number_of_nodes_in_layer]))
	b_2 = tf.Variable(tf.zeros([number_of_nodes_in_layer]))
	layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
	layer_2 = tf.nn.relu(layer_2)

	# layer 2 multiplying and adding bias then activation function
	W_O = tf.Variable(tf.random_uniform([number_of_nodes_in_layer, 1]))
	b_O = tf.Variable(tf.zeros([1]))
	output = tf.add(tf.matmul(layer_2,W_O), b_O)

	# O/p layer multiplying and adding bias then activation function
	# notice output layer has one node only since performing #regression
	return output


if __name__ == '__main__':

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
	y_train = scaler.fit_transform(df_train['Close'].as_matrix().reshape(-1, 1))

	# y is output and x is features
	X_test = scaler.fit_transform(df_test.drop(['Close'], axis=1).as_matrix())
	y_test = scaler.fit_transform(df_test['Close'].as_matrix().reshape(-1, 1))

	# tf.placeholder() will define gateway for data to graph
	xs = tf.placeholder("float")
	ys = tf.placeholder("float")

	output = neural_net_model(xs, 3)

	# Our mean squared error cost function
	cost = tf.reduce_mean(tf.square(output-ys))

	# Gradient Descent optimiztion for updating weights and biases
	train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

	# Cost of training at each iteration
	c_t = []
	# Cost of testing at each iteration
	c_test = []

	with tf.Session() as sess:
		# Initiate session and initialize all vaiables
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		for i in range(100):
			for j in range(X_train.shape[0]):
				# Run cost and train with each sample
				sess.run([cost,train],feed_dict={xs:X_train[j,:].reshape(1,3), ys:y_train[j]})

			c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
			c_test.append(sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
			print('Epoch :',i,'Cost :',c_t[i])

		# Predict output of test data after training
		pred = sess.run(output, feed_dict={xs:X_test})

		print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))

		# Show results
		plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
		plt.plot(range(y_test.shape[0]),pred,label="Predicted Data")
		plt.legend(loc='best')
		plt.ylabel('Stock Value')
		plt.xlabel('Days')
		plt.title('Stock Market Nifty')
		plt.show()

		if raw_input('Save model ? [Y/N]: ') == 'Y':
			# Save model
			saver.save(sess,'./yahoo_dataset.ckpt')
			print('Model Saved')

