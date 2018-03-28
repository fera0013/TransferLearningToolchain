# Transfer learning with Tensorflow Object Detection API

Transfer learning is the adaption of pre-trained models to similar or moderately different tasks, by fine-tuning parameters of the pre-trained models. Using the transfer learning approach can help you to develop powerful models, by building on the results of top experts in the deep learning field. Although it sounds like a simple task, transfer learning still requires a lot of [research](https://machinelearningmastery.com/transfer-learning-for-deep-learning/), thorough preparation, development and testing. 

This repository aims to provide a template to help you mastering the technological part of transfer learning for object detection tasks, using the powerful [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). If you know what you want to use transfer learning for, follow the steps below to get a working implementation. 

## Dataset

A typical application for transfer learning is the adaptions of existing models, to detect new object classes not contained in the dataset the models were originally trained on. Depending on the similarity of the original and the new object classes, different parts of models have to be fine-tuned. A necessary prerequisite for that is to obtain or generate sample images, representing the new object classes.

### Generate a new dataset (optional) 



