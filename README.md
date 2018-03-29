# A Transfer Learning Toolchain for Semantic Segmentation

Transfer learning is the adaption of pre-trained models to similar or moderately different tasks, by finetuning parameters of the pre-trained models. Using the transfer learning approach can help you to develop powerful models, by building on the results of top experts in the deep learning field. Although it sounds like a simple task, transfer learning still requires a lot of [research](https://machinelearningmastery.com/transfer-learning-for-deep-learning/), thorough preparation, development and testing. 

This repository aims to provide a toolchain covering the mere technical aspects of transfer learning for semantic segmentation. The instructions below follow an exemplary path to a succesful transfer learning model, based on a specific combination of tools, frameworks and models. We use

* [google-image-downloader](https://github.com/fera0013/google-images-download), to create a csv with image URLs, based on google search queries
* [https://www.labelbox.io/](https://www.labelbox.io/), to label, export and convert the dataset
* one of the [models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models) trained on the [COCO dataset](http://cocodataset.org/#home)
* [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), for transfer learning

There are no doubt many alternatives for each step in this toolchain and the approach to transfer learning varies greatly, depending on the particular task. Use this toolchain if you want to get started relatively quickly and try to adapt the individual steps to your specific requirements.

## Dataset generation

A typical application for transfer learning is the adaption of existing models, to detect new object classes not contained in the dataset the models were originally trained on. Depending on the similarity of the original and the new object classes, different parts of the pretrained models have to be finetuned. A necessary prerequisite for that is to obtain or generate sample images, representing the new object classes. In our examplary workflow, we want to use a model pre-trained on the [COCO dataset](http://cocodataset.org/#home), to detect waste bottles. 

### Generate a new dataset using labelbox 

If you can't find an existing dataset representing the novel objects you want to detect, you need to collect and label images yourself. Several tools exist which greatly simplify the painstaking process of generating new datasets. The following steps roughly describe the workflow using labelbox [https://www.labelbox.io/](https://www.labelbox.io/), a powerful cloud based labeling tool, with an easy to use interface. For  more detailed instructions, go to the [labelbox documentation](https://github.com/Labelbox/Labelbox).

1. Create a csv-file with the URLs of images, representing your novel classes
 Use [this script](https://github.com/fera0013/google-images-download), if you want to automize the process based on google search queries. 
2.  [Create a labelbox project and upload your dataset](https://github.com/Labelbox/Labelbox#quickstart)
3. (optional) [Adapt the labeling interface](https://github.com/Labelbox/Labelbox#creating-custom-labeling-interfaces) 
4. Label away using the [semantic segmentation interface](https://github.com/Labelbox/Labelbox#image-segmentation-interface) 
5. [Export the labeled dataset](https://github.com/Labelbox/Labelbox#exporting-labels) in json format
6. Use [this script](https://github.com/Labelbox/Labelbox/blob/master/scripts/README.md#labelbox-json-to-coco) to convert from json to COCO format.
7. Place the output file from step 6 in the [data folder](data/) (See the [sample file](https://github.com/fera0013/TransferLearningWithTensorflowAPI/blob/master/data/coco_labels.json) for a reference of the expected output format)

## Transfer learning with [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

### Convert the COCO labels to TFRecords 
[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) requires the data to be in the somewhat obscure [TFRecord](https://www.tensorflow.org/programmers_guide/datasets) format. [Understanding the TFRecord format](https://planspace.org/20170323-tfrecords_for_humans/) and getting it right is not an easy task and may take some time. Fortunately, the tensorflow records provides some scripts for the most common formats, such as [coco to TFRecord conversion](https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py). We have slightly adapted this script to download the sample images based on the URLs contained in the [labelbox COCO export](https://github.com/fera0013/TransferLearningWithTensorflowAPI/blob/master/data/coco_labels.json).

To create the TFRecords

1. Open a command line and cd to the [script folder](data/script)
2. Enter 
```
python create_coco_tf_record.py --train_annotations_file=../data/coco_labels.json --val_annotations_file=../data/coco_labels.json --testdev_annotations_file=../data/coco_labels.json   --train_image_dir=../data/images/train --val_image_dir=../data/images/val --test_image_dir=../data/images/test --output_dir=../data
```

3. Navigate to [the data folder](data/), where you should find 3 .record files if the script completed correctly

Please note that - for convenience - we used the same annotation file for training, validation and testing. For production, you should use disjoint annotation files for each of these tasks. 

### Transfer learn the new classes

The following steps very much depend on many different aspects, such as the model you intend to use and the relation between the new classes and the classes the model was originally trained on. In our example, we train [one of the models pretrained on the COCO dataset](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models) to detect waste bottles, which are closely related to the bottles class contained in the original [COCO dataset](http://cocodataset.org/).

1. Download [one of the models pretrained on the COCO dataset](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models-coco-models)
2. Unpack the downloaded tar
3. Copy all three files with .ckpt.* ending to the [model folder](model/)
3. Copy the config file for the model you want to use from the [tensorflow repository](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) and save it to the [model folder](model/).
4. Create a `.pbtxt` file with the label mappings (see [data/label_map.pbtxt](this sample file) for a reference)
5. open the config file and adapt the following parts (see [data/faster_rcnn_resnet50_coco.config](for a reference config):
* Change the `fine-tune checkpoint` entry to  `fine_tune_checkpoint: "../model/model.ckpt""`
* Change the `input_path` value of the `train_input_reader` entry to  `input_path: "../data/coco_train.record"`
* Change `label_map_path` value of the `train_input_reader` entry to `label_map_path: "../data/label_map.pbtxt"`
* Change the `input_path` value of the `val_input_reader` entry to  input_path: `"../data/coco_val.record"`
* Change `label_map_path` value of the `val_input_reader` entry to `label_map_path: "../data/label_map.pbtxt"`
7. Change other entries according to your requirements
8. open a command line and cd to the [script/] folder
9. enter 
```
python train.py --logtostderr --train_dir=../model/train --pipeline_config_path=../model/faster_rcnn_resnet50_coco.config
```
If everything was configured correctly, the training should complete succesfully.

### Test the new model

To be done...
