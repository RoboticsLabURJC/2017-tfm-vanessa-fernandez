---
layout: default
---
# Week 4: Starting with DetectionSuite

[DeepLearning Suite](https://github.com/JdeRobot/DeepLearningSuite) is a set of tool that simplify the evaluation of most common object detection datasets with several object detection neural networks. It offers a generic infrastructure to evaluates object detection algorithms againts a dataset and compute most common statistics: precision, recall. DeepLearning Suite supports YOLO (darknet) and Background substraction. I've installed YOLO(darknet). I follow the following steps: 

<pre>
git clone https://github.com/pjreddie/darknet
cd darknet
make
</pre>

Downloading the pre-trained weight file: 

<pre>
wget https://pjreddie.com/media/files/yolo.weights
</pre>

Then run the detector: 

<pre>
./darknet detect cfg/yolo.cfg yolo.weights data/dog.jpg
</pre>

