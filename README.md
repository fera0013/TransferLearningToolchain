# Transfer learning with Tensorflow Object Detection API

Transfer learning is the adaption of pre-trained models to similar or moderately different tasks, by fine-tuning parameters of the pre-trained models. Using the transfer learning approach can help you to develop powerful models, by building on the results of top experts in the deep learning field. Although it sounds like a simple task, transfer learning still requires a lot of [research](https://machinelearningmastery.com/transfer-learning-for-deep-learning/), thorough preparation, development and testing. 

This repository aims to provide a template to help you mastering the technological part of transfer learning for object detection tasks, using the powerful [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). If you know what you want to use transfer learning for, follow the steps below to get a working implementation. 

## Dataset

A typical application for transfer learning is the adaption of existing models, to detect new object classes not contained in the dataset the models were originally trained on. Depending on the similarity of the original and the new object classes, different parts of models have to be fine-tuned. A necessary prerequisite for that is to obtain or generate sample images, representing the new object classes.

### Generate a new dataset using labelbox (optional) 

If you can't find an existing dataset covering the objects you want to detect, you need to collect and label images yourself. Several tools exist which greatly simplify the painstaking process of generating new datasets. The following steps roughly describe the workflow using labelbox [https://www.labelbox.io/](https://www.labelbox.io/), a powerful cloud based labeling tool, with an easy to use interface. For  more detailed instructions, go to the [labelbox documentation](https://github.com/Labelbox/Labelbox).

1. Create a csv-file with the URLs of images, representing your novel classes
 Use [this script](https://github.com/fera0013/google-images-download), if you want to automize the process based on google search queries. 
2.  [Create a labelbox project and upload your dataset](https://github.com/Labelbox/Labelbox#quickstart)
3. (optional) [Adapt the labeling interface] (https://github.com/Labelbox/Labelbox#creating-custom-labeling-interfaces) 
4. Label away
5. [Export the labeled dataset](https://github.com/Labelbox/Labelbox#exporting-labels) in json format
6. Use [this script](https://github.com/Labelbox/Labelbox/blob/master/scripts/README.md#labelbox-json-to-coco) to convert from json to COCO format.

