import keras
from keras.layers import Input ,Dense, Dropout, Activation, LSTM
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model

import numpy as np

timesteps = 100
number_of_samples = 250
nb_samples = number_of_samples
frame_row = 32
frame_col = 32
channels = 3

nb_epoch = 1
batch_size = timesteps

data = np.random.random((250, timesteps, frame_row, frame_col, channels))
label = np.random.random((250, 1))

X_train = data[0:200,:]
y_train = label[0:200]

X_test = data[200:,:]
y_test = label[200:,:]


# Define network architecture and compile
model = Sequential()                        

model.add(TimeDistributed(Convolution2D(32, (3, 3), border_mode='same'), input_shape=X_train.shape[1:]))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Convolution2D(32, (3, 3))))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.25)))

model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(512)))
                
                
model.add(TimeDistributed(Dense(35, name="first_dense" )))
        
model.add(LSTM(20, return_sequences=True, name="lstm_layer"));
         
model.add(TimeDistributed(Dense(1), name="time_distr_dense_one"))
model.add(GlobalAveragePooling1D(name="global_avg"))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# We train the model
model.fit(X_train, y_train, batch_size=16, epochs=15) 

# We evaluate the model
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

