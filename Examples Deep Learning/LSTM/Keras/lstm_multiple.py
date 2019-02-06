"""

Example of LSTM with Multiple Input Features
Code from: https://github.com/keras-team/keras/issues/4870

"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# This data can be framed as 1 sample with 9 time steps and 11 features
# It can be reshaped as a 3D array as follows
# The LSTM input layer must be 3D. The meaning of the 3 input dimensions are: samples, time steps, and features.
# The LSTM input layer is defined by the input_shape argument on the first hidden layer.
# The input_shape argument takes a tuple of two values that define the number of time steps and features.
# The number of samples is assumed to be 1 or more.
# The reshape() function on NumPy arrays can be used to reshape your 1D or 2D data to be 3D.

# Input sequence
wholeSequence = [[0,0,0,0,0,0,0,0,0,2,1],
                 [0,0,0,0,0,0,0,0,2,1,0],
                 [0,0,0,0,0,0,0,2,1,0,0],
                 [0,0,0,0,0,0,2,1,0,0,0],
                 [0,0,0,0,0,2,1,0,0,0,0],
                 [0,0,0,0,2,1,0,0,0,0,0],
                 [0,0,0,2,1,0,0,0,0,0,0],
                 [0,0,2,1,0,0,0,0,0,0,0],
                 [0,2,1,0,0,0,0,0,0,0,0],
                 [2,1,0,0,0,0,0,0,0,0,0]]

# Preprocess Data:
wholeSequence = np.array(wholeSequence, dtype=float) # Convert to NP array.
data = wholeSequence[:-1] # all but last
target = wholeSequence[1:] # all but first

print(data)
print(data.shape)
print(target)
print(target.shape)

# Reshape training data for Keras LSTM model
# The training data needs to be (batchIndex, timeStepIndex, dimentionIndex)
# Single batch, 9 time steps, 11 dimentions
data = data.reshape((1, 9, 11))
target = target.reshape((1, 9, 11))

print('Reshape', data.shape)
print('Reshape', target.shape)


# Build Model
model = Sequential()  
model.add(LSTM(11, input_shape=(9, 11), unroll=True, return_sequences=True))
model.add(Dense(11))
model.compile(loss='mae', optimizer='adam', metrics=['accuracy', 'mse'])
model.fit(data, target, epochs=2000, batch_size=1, verbose=2)

score = model.evaluate(data, target)
print('Test mae:', score[0])
print('Test accuracy:', score[1])
print('Test mse:', score[2])

