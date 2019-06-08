---
layout: default
---
# Week 2: BBDD of Deep Learning and C++ Tutorials

## BBDD of Deep Learning

This second week, I've known some datasets used in Deep Learning. Datasets provide a means to train and evaluate algorithms, they drive research in new directions. Datasets related to object recognition can be split into three groups: object classification, object detection and semantic scene labeling.

*Object classification*: assigning pixels in the image to categories or classes of interest. There are different datasets for image classification. Some of them are:

* MNIST (Modified National Institute of Standards and Technology): is a large database of handwritten digits that is commonly used for training various image processing systems. It consists of a set training of 60,000 examples and a test of 10,000 examples. Is a good database for people who want to try learning techniques and methods of pattern recognition in real-world data, while dedicating a minimum effort to preprocess and format.

* Caltech 101: is intended to facilitate Computer Vision research and techniques and is most applicable to techniques involving image recognition classification and categorization. Caltech 101 contains a total of 9,146 images, split between 101 distinct object categories and a background category. Provided with the images are a set of annotations describing the outlines of each image, along with a Matlab script for viewing.

* Caltech 256: is a set similar to the Caltech 101, but with some improvements: 1) the number of categories is more than doubled, 2) the minimum number of images in any category is increased from 31 to 80, 3) artifacts due to image rotation are avoided, and 4) a new and larger clutter category is introduced for testing background rejection.

* CIFAR-10: is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. There are 50,000 training images and 10,000 test images in the official data.

* CIFAR-100: is large, consisting of 100 image classes, with 600 images per class. Each image is 32x32x3 (3 color), and the 600 images are divided into 500 training, and 100 test for each class.

* ImageNet: is organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a "synonym set" or "synset". There are more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). ImageNet provide on average 1000 images to illustrate each synset. Images of each concept are quality-controlled and human-annotated.


*Object detection*: the process of finding instances of real-world objects in images or videos. Object detection algorithms usually use extracted features and learning algorithms to recognize instances of an object category. There are different datasets for object detection. Some of them are:

* COCO (Microsoft Common Objects in Context): is an image dataset designed to spur object detection research with a focus on detecting objects in context. COCO has the following characteristics: multiple objects in the images, more than 300,000 images, more than 2 million instances, and 80 object categories. Training sets, test and validation are used with their corresponding annotations. The annotations include pixel-level segmentation of object belonging to 80 categories, keypoint annotations for person instances, stuff segmentations for 91 categories, and five image captions per image.

* PASCAL VOC: In 2007, the group has two large databases, one of which consists of a validation set and another of training, and the other with a single test set. Both databases contain about 5000 images in which they are represented, approximately 12,000 different objects, so, in total, this set has of about 10,000 images in which about 24,000 objects are represented. In the year 2012, this setis modified, increasing the number of images with representation to 11530 of 27450 different objects.

* Caltech Pedestrian Dataset: consists of approximately 10 hours of 640x480 30Hz video taken from a vehicle driving through regular traffic in an urban environment. About 250,000 frames with a total of 350,000 bounding boxes and 2300 unique pedestrians were annotated. The annotation includes temporal correspondence between bounding boxes and detailed occlusion labels.

* TME Motorway Dataset (Toyota Motor Europe (TME) Motorway Dataset): is composed by 28 clips for a total of approximately 27 minutes (30000+ frames) with vehicle annotation. Annotation was semi-automatically generated using laser-scanner data. The dataset is divided in two sub-sets depending on lighting condition, named “daylight” (although with objects casting shadows on the road) and “sunset” (facing the sun or at dusk). 


*Semantic scene labeling*: each pixel of an image have to be labeled as belonging to a category. There are different dataset for semantic scene labeling. Some of them are:

*  SUN dataset: provide a collection of annotated images covering a large variety of environmental scenes, places and the objects within. SUN contains 908 scene categories from the WordNet dictionary with segmented objects. The 3,819 object categories span those common to object detection datasets (person, chair, car) and to semantic scene labeling (wall, sky, floor). There are a few categories have a large number of instances (wall: 20,213, window: 16,080, chair: 7,971) while most have a relatively modest number of instances (boat: 349, airplane: 179, floor lamp: 276).

* Cityscapes Dataset: contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5 000 frames in addition to a larger set of 20 000 weakly annotated frames. 



## C++ Tutorials

In this week, I've been doing c ++ tutorials. I followed the tutorials on the next [page](https://codigofacilito.com/cursos/c-plus-plus). 

