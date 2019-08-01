---
layout: default
---
# Week 33: Controlnet, Reading information

## Controlnet

[Reactive Ground Vehicle Control via Deep Networks](https://pdfs.semanticscholar.org/ec17/ec40bb48ec396c626506b6fe5386a614d1c7.pdf) present a deep learning based reactive controller that uses a simple network architecture (ControlNet) requiring few training images. ControlNet has 63223 trainable parameters in the following structure: 

<pre>
* 2D Convolution, 16 filters of size 10x10
* Max Pooling, filter size 3x3, stride of 2
* 2D Convolution, 16 filters of size 5x5
* Max Pooling, filter size 3x3, stride of 2
* 2D Convolution, 16 filters of size 5x5
* Max Pooling, filter size 3x3, stride of 2
* 2D Convolution, 16 filters of size 5x5
* Max Pooling, filter size 3x3, stride of 2
* 2D Convolution, 16 filters of size 5x5
* Max Pooling, filter size 3x3, stride of 2
* Fully connected, 50 neurons
* ReLu
* Fully connected, 50 neurons
* ReLu
* LSTM (5 frames)
* Softmax with 3 ouputs
</pre>


![controlnet_architecture](https://roboticslaburjc.github.io/2017-tfm-vanessa-fernandez/images/controlnet_architecture.png)


I've adapted the model to my data:

<pre>
model = Sequential()
model.add(Conv2D(16, (5, 5), input_shape=img_shape, activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(16, (5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(50, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Flatten())
model.add(Reshape((100, 1)))
model.add(LSTM(5))
model.add(Activation('softmax'))
model.add(Dense(1))
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
</pre>


|            Driving results (Controlnet, whole image)                  |||||
|                           |        Manual        ||      ControlNet      || 
|      Circuits             | Percentage |   Time   | Percentage |   Time   | 
|  Simple (clockwise)       |    100%    | 1min 35s |     100%   | 1min 46s |
|Simple (anti-clockwise)    |    100%    | 1min 32s |     100%   | 1min 46s |
|  Monaco (clockwise)       |    100%    | 1min 15s |      45%   |          |
|Monaco (anti-clockwise)    |    100%    | 1min 15s |      10%   |          |
| Nurburgrin (clockwise)    |    100%    | 1min 02s |     100%   | 1min 05s |
|Nurburgrin (anti-clockwise)|    100%    | 1min 02s |      92%   |          |
|   CurveGP (clockwise)     |    100%    | 2min 13s |     100%   | 2min 26s |
| CurveGP (anti-clockwise)  |    100%    | 2min 09s |      75%   |          |
|   Small (clockwise)       |    100%    | 1min 00s |     100%   | 1min 01s |
| Small (anti-clockwise)    |    100%    |    59s   |     100%   |    59s   |




## Reading information

This week I was reading more thoroughly LSTM networks. I used [1](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) for a better understanding of this kind of networks.

Also I've followed the tutorial [Using a Keras Long Short-Term Memory (LSTM) Model to Predict Stock Prices](https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html). 



### Long-term Recurrent Convolutional Networks

In this [paper](https://arxiv.org/pdf/1411.4389.pdf) recurrent sequence models are directly connected to modern visual convolutional networkmodels and can be jointly trained to learn temporal dynamics and convolutional perceptual representations. In this paper, they propose Long-term Recurrent Convolutional Networks(LRCNs), a class of architectures which combines convolutional layers and long-range temporal recursion and is end-to-end trainable.

They show here that convolutional networks with recurrent units are generally applicable to visual time-series modeling, and argue that in visual tasks where static or flat temporal models have previously been employed, LSTM-style RNNs can provide significant improvement when ample training data are available to learn or refine the representation. They also show that these models improve generation of descriptions from intermediate visualrepresentations derived from conventional visual models.

This work proposes a Long-term Recurrent ConvolutionalNetwork (LRCN) model combining a deep hierarchical visual feature extractor (such as a CNN) with a model that can learn to recognize and synthesize temporal dynamics for tasks involving sequential data (inputs or outputs), visual,linguistic, or otherwise. 


