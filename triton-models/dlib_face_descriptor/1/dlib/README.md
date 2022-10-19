# dlib-models

[This repository](https://github.com/davisking/dlib-models) contains trained models created by ([Davis King](https://github.com/davisking)). They are provided as part of the dlib example programs, which are intended to be educational documents that explain how to use various parts of the dlib library. As far as I am concerned, anyone can do whatever they want with these model files as I've released them into the public domain. Details describing how each model was created are summarized below.

## dlib_face_recognition_resnet_model_v1.dat.bz2

This model is a ResNet network with 29 conv layers. It's essentially a version of the ResNet-34 network from the paper Deep Residual Learning for Image Recognition by He, Zhang, Ren, and Sun with a few layers removed and the number of filters per layer reduced by half.

The network was trained from scratch on a dataset of about 3 million faces. This dataset is derived from a number of datasets. The face scrub dataset (http://vintage.winklerbros.net/facescrub.html), the VGG dataset (http://www.robots.ox.ac.uk/~vgg/data/vgg_face/), and then a large number of images I scraped from the internet. I tried as best I could to clean up the dataset by removing labeling errors, which meant filtering out a lot of stuff from VGG. I did this by repeatedly training a face recognition CNN and then using graph clustering methods and a lot of manual review to clean up the dataset. In the end about half the images are from VGG and face scrub. Also, the total number of individual identities in the dataset is 7485. I made sure to avoid overlap with identities in LFW.

The network training started with randomly initialized weights and used a structured metric loss that tries to project all the identities into non-overlapping balls of radius 0.6. The loss is basically a type of pair-wise hinge loss that runs over all pairs in a mini-batch and includes hard-negative mining at the mini-batch level.

The resulting model obtains a mean error of 0.993833 with a standard deviation of 0.00272732 on the LFW benchmark.
