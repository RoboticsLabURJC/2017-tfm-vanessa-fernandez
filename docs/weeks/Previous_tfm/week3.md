---
layout: default
---
# Week 3: Getting started

## Install Keras

Keras is a high-level neural networks library, written in Python and capable of running on top of either TensorFlow or Theano (they're both open source libraries for numerical computation optimized for GPU and CPU). It was developed with a focus on enabling fast experimentation. I followed the next steps to install Keras: [1](https://keras.io/#installation). Currently, I'm running Keras on top of [TensorFlow](https://www.tensorflow.org/) optimized for CPU.

Keras includes a module for multiple supplementary tasks called Utils. The most important functionality for the project provided by this module is the .HDF5Matrix() method. Keras employs the [HDF5 file format](http://docs.h5py.org/en/latest/build.html) to save models and read datasets. According to HDF5 documentation, it is a hierarchical data format designed for high volumes of data with complex relationships.

The commands to have installed keras and HDF5 are: 

<pre>
sudo pip install tensorflow
sudo pip install keras
sudo apt-get install libhdf5
sudo pip install h5py
</pre>


## Testing the code of a neural network

After installing Keras, I've tried the tfg of my mate [David Pascual](http://jderobot.org/Dpascual-tfg). In his project, he was be able to classify a well-known database of handwritten digits (MNIST) using a convolutional neural network (CNN). He created a program that is able to get images from live video and display the predictions obtained from them. I've studied his code and I tested its operation.

If you want to execute David's code you have to open two terminals and put in each one respectively: 

<pre>
cameraserver cameraserver_digitclassifier.cfg 
</pre>

<pre>
python digitclassifier.py digitclassifier.cfg
</pre>

![digit](http://jderobot.org/store/vmartinezf/uploads/images/digit.png)


## First neural network: MNIST example

After studying David's code a bit, I have trained a simple convolutional neural network with the MNIST dataset.

First of all, we have to load and adapt the input data. Keras library contains a module named datasets from which we can import a few databases, including MNIST. In order to load MNIST data, we call mnist.load_data() function. It returns images and labels from both training and test datasets (x_train, y train and x_test, y_test respectively). 

<pre>
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
</pre>

The dimension of x_train are: (6000, 28, 28). That is, we have 6000 images with a size of 28x28. We also have to explicitly declare the depth of the samples. In this case, we're working with black and white images, so depth will be equal to 1. We reshape data using reshape() method. Depending on the backend (TensorFlow or Theano), the arguments must be passed in different order. 

<pre>
# Shape of image depends if you use TensorFlow or Theano
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
</pre>


The next step is to convert data type from uint8 to float32 and normalize pixel values to [0,1] range. 

<pre>
# We normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
</pre>

<pre>
('x_train shape:', (60000, 28, 28, 1))
(60000, 'train samples')
(10000, 'test samples')
</pre>

We have to reshape label data using utils.to_categorical() method. 

<pre>
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
<pre>

Now that data is ready, we have to define the architecture of our neural network. We're going to define the model architecture that we will use. We use sequential model. 

<pre>
model = Sequential()
</pre>

Next step is to add input, inner and output layers. We can add layers using the add(). Convolutional neural networks usually contain: convolutional layer, pooling layer and fully connected layer. We're going add these layers to our model. 

<pre>
model.add(Conv2D(nb_filters, kernel_size=nb_conv,
                     activation='relu',
                     input_shape=input_shape))
model.add(Conv2D(64, nb_conv, activation='relu'))
model.add(MaxPooling2D(pool_size=nb_pool))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
</pre>

We have to train the neural network. We must use the fit() method. 

<pre>
# We train the model
history = model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, verbose=1,
          validation_data=(x_test, y_test))
</pre>

Last step is to evaluate the model. 

<pre>
# We evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
</pre>

Keras displays the results obtained. 

<pre>
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
  128/60000 [..............................] - ETA: 4:34 - loss: 2.2873 - acc: 0 
256/60000 [..............................] - ETA: 4:12 - loss: 2.2707 - acc: 0 
384/60000 [..............................] - ETA: 3:59 - loss: 2.2541 - acc: 0 
512/60000 [..............................] - ETA: 3:51 - loss: 2.2271 - acc: 0 
640/60000 [..............................] - ETA: 3:47 - loss: 2.1912 - acc: 0

...

59520/60000 [============================>.] - ETA: 1s - loss: 0.0382 - acc: 0.988
59648/60000 [============================>.] - ETA: 1s - loss: 0.0382 - acc: 0.988
59776/60000 [============================>.] - ETA: 0s - loss: 0.0382 - acc: 0.988
59904/60000 [============================>.] - ETA: 0s - loss: 0.0381 - acc: 0.988
60000/60000 [==============================] - 264s 4ms/step - loss: 0.0381 - acc: 0.9889 - val_loss: 0.0293 - val_acc: 0.9905
('Test loss:', 0.029295223668735708)
('Test accuracy:', 0.99050000000000005)
</pre>


The fit() function returns a history object which has a dictionary of all the metrics which were required to be tracked during training. We can use the data in the history object to plot the loss and accuracy curves to check how the training process went. You can use the history.history.keys() function to check what metrics are present in the history. It should look like the following [‘acc’, ‘loss’, ‘val_acc’, ‘val_loss’]. 


![Loss_Curves](http://jderobot.org/store/vmartinezf/uploads/images/Loss_Curves.png)

![Accuracy_Curves](http://jderobot.org/store/vmartinezf/uploads/images/Accuracy_Curves.png)


