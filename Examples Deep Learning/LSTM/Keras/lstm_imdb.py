"""

LSTM example
Code from tutorial: https://rubikscode.net/2018/03/26/two-ways-to-implement-lstm-network-using-python-with-tensorflow-and-keras/
Dataset: https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification

"""

from keras.preprocessing import sequence 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Embedding, LSTM 
from keras.datasets import imdb

# We load dataset of top 1000 words
num_words = 1000 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

# We need to divide this dataset and create and pad sequences (using sequence from keras.preprocessing)
# In the padding we used number 200, meaning that our sequences will be 200 words long
X_train = sequence.pad_sequences(X_train, maxlen=200) 
X_test = sequence.pad_sequences(X_test, maxlen=200)

print(X_train)
print(X_test)
print(X_train.shape)
print(X_test.shape)

# Define network architecture and compile 
model = Sequential() 
model.add(Embedding(num_words, 50, input_length=200)) 
model.add(Dropout(0.2)) 
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
model.add(Dense(250, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

# We train the model
model.fit(X_train, y_train, batch_size=64, epochs=15) 

# We evaluate the model
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# We save the model
#model.save('model_imdb.h5')
