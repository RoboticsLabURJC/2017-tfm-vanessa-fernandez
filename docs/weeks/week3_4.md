---
layout: default
---
# Week 3, 4: Understanding LSTM

## Understanding LSTM

Sometimes it is necessary to use previous information to process the current information. Traditional neural networks can not do this, and it seems an important deficiency. Recurrent neural networks address this problem. They are networks with loops that allow the information to persist. A recurrent neural network can be thought of as multiple copies of the same network, each passing a message to a successor. In the last few years, there have been incredible success applying RNNs to a variety of problems: speech recognition, language modeling, translation, image captioning ... Essential to these successes is the use of “LSTMs,” a very special kind of recurrent neural network which works, for many tasks, much much better than the standard version.

One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task. Sometimes, we only need to look at recent information to perform the present task. For example, consider a language model trying to predict the next word based on the previous ones. If we are trying to predict the last word in “the clouds are in the sky,” we don’t need any further context – it’s pretty obvious the next word is going to be sky. In such cases, where the gap between the relevant information and the place that it’s needed is small, RNNs can learn to use the past information.

But there are also cases where we need more context. Consider trying to predict the last word in the text “I grew up in France… I speak fluent French.” Recent information suggests that the next word is probably the name of a language, but if we want to narrow down which language, we need the context of France, from further back. It’s entirely possible for the gap between the relevant information and the point where it is needed to become very large. Unfortunately, as that gap grows, RNNs become unable to learn to connect the information.

In theory, RNNs are absolutely capable of handling such “long-term dependencies”. In practice, RNNs don’t seem to be able to learn them. LSTMs don’t have this problem.

LSTMs (Long Short-Term Memory networks) ([1](http://www.bioinf.jku.at/publications/older/2604.pdf), [2](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)) are a type of RNN (Recurrent Neural Network) architecture that addresses the vanishing/exploding gradient problem and allows learning of long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997). They work very well on a large variety of problems, and are now widely used.

LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior. All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure. LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

![LSTM](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/lstm.png)

In the above diagram, each line carries an entire vector, from the output of one node to the inputs of others. The pink circles represent pointwise operations, like vector addition, while the yellow boxes are learned neural network layers. Lines merging denote concatenation, while a line forking denote its content being copied and the copies going to different locations.

The main idea consists in a memory cell as a interchangeably block which can maintain its state over time. The key to LSTMs is the cell state (the horizontal line running through the top of the diagram). The cell state runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.

The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates. Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a pointwise multiplication operation. The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through. A value of zero means “let nothing through,” while a value of one means “let everything through”. An LSTM has three of these gates, to protect and control the cell state.


The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer". The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, that could be added to the state. In the next step, we’ll combine these two to create an update to the state.

It’s now time to update the old cell state, Ct−1, into the new cell state Ct. Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanh (to push the values to be between −1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to. 


## First exampke of LSTM

To implement a first simple example of this kind of networks a [tutorial](https://www.knowledgemapper.com/knowmap/knowbook/jasdeepchhabra94@gmail.comUnderstandingLSTMinTensorflow(MNISTdataset)) has been followed in which we discover how to develop an LSTM network in tensorflow. In this example, we use MNIST as our dataset. The result of this implementation can be found in [Github](https://github.com/RoboticsURJC-students/2017-tfm-vanessa-fernandez/blob/master/Examples%20Deep%20Learning/LSTM/Tensorflow/lstm_mnist.py). 

