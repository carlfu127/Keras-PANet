# Keras-PANet
PANet for Instance Segmentation and Object Detection 

This is an implementation of [PANet](https://arxiv.org/abs/1803.01534) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet backbone.
# Training on MS COCO
In this repository, i tarin the code with BN layers in the backbone fixed and use GN in other part. Training and evaluation code is in `samples/coco/coco.py`.
