Birds Species classification using transfer learning
====================

Introduction
~~~~~~~~~~~~

This tutorial explains how to use 'rmldnn' to perform transfer learning to train a model that classifies birds species images from dataset.(https://www.kaggle.com/datasets/gpiosenka/100-bird-species).

Starting with a brief introduction about Transfer Learning, Transfer learning is a machine learning method in which a model generated for one job is reused as the starting point for a model on a different task. Here we have leveraged pre-trained RESNET50, which is trained on more than a million images from the ImageNet database. RESNET50 is CNN (Convolutional Neural Network) model which is about 50 layers deep. Below Image shows architecture of RESNET50 

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/images/ResnetArch.png?raw=true
    :width: 750
    :align: center
  
The above tasks will exemplify how to use `rmldnn` to:

 - perform `transfer learning` by reusing model weights in a different model;
 - train a convolutional neural network;
 - use learning rate scheduler to decay leraning rate to reach to optimal model quickly.


The dataset
~~~~~~~~~~~

We will use Kaggle Birds 400 Database which contains 400 bird species.58388 training images, 2000 test images(5 images per species) and 2000 validation images(5 images per species.The classes includes ABBOTTS BABBLER, ABBOTTS BOOBY, ABYSSINIAN GROUND HORNBILL and 397 other birds specieis. You can get this dataset from  `here <https://www.kaggle.com/datasets/gpiosenka/100-bird-species>`__ (as a zip file)

Note: There is an error in the training set in the directory BLACK & YELLOW BROADBILL in the dataset provided by Kaggle, which contains an extra space that is not present in the validation or testing sets. Please rename this file in the training set to remove any unnecessary space before using it.

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/images/birds_cover.jpg?raw=true

The pre-processed dataset can be downloaded directly from here for convenience.

On unzipping downloaded file we'll assume that it has following directory structure:

.. code:: bash

    +-- Data/
    |   +-- train/
        |   +-- ABBOTTS BABBLER/
        |   +-- ABBOTTS BOOBY/
        |   +-- ...
        |   +-- YELLOW HEADED BLACKBIRD/
    |   +-- valid/
        |   +-- ABBOTTS BABBLER/
        |   +-- ABBOTTS BOOBY/
        |   +-- ...
        |   +-- YELLOW HEADED BLACKBIRD/
    |   +-- test/
        |   +-- ABBOTTS BABBLER/
        |   +-- ABBOTTS BOOBY/
        |   +-- ...
        |   +-- YELLOW HEADED BLACKBIRD/


The images are multi-channel (Coloured) with size of 224 X 224, similar to ones in the figure below. 

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/images/Birds_joined.png?raw=true

The neural network
~~~~~~~~~~~~~~~~~~

Since we'll be doing transfer learning, we'll need to first get our base model, which in our instance is RESNET50, and then add a single 400-unit dense layer at the end (with a log-softmax activation). After that, we'll need to save our prepared model as a Hdf5 file and our network architecture as a .json file so that we can train it with rmldnn. The network is depicted and described below:

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/images/network_arch.png?raw=true
    :height: 500
    :align: center

You can perform following steps to obtain model and its architecture. We've additionally made .hdf5 to download directly from  `here <https://rmldnnstorage.blob.core.windows.net/rmldnn-models/model_resnet50_imagenet.h5>`__.

.. code:: bash

    #importing libraries
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50
    #preparing base model(RESNEt50)
    pretrained_model = ResNet50(
        input_shape=(224,224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    pretrained_model.trainable = False
    #adding dense layer with 400 units, log_softmax activation after base model
    inputs = pretrained_model.input
    outputs = Dense(400, activation='log_softmax')(pretrained_model.output)
    model = Model(inputs=inputs, outputs=outputs)
    #saving model
    model.save("model.h5")
    #saving architecture in json format
    d=model.to_json()
    with open("layers.json",'w') as f:
        f.write(d)
        
 Note: After getting layers.json file from above steps kindly edit it using editor of choice and change activation function of last layer from "log_softmax_v2" to "log_softmax" cause it may lead to an error.

Running training
~~~~~~~~~~~~~~~~

`rmldnn` is a code-free, high-performance tool for distributed deep-learning, and the entire flow can be defined
in a single configuration file. To perform transfer learning using rmldnn we first need to load our prepared model as well as we will also use our network architecture json file which tells about layers present in our model. We will assume following directory structure is maintained inside main folder:


.. code:: bash

    +-- birds_classification/
    |   +-- data/
        |   +-- train/
        |   +-- test/
        |   +-- valid/
    |   +-- model_checkpoint/
        |   +-- model.h5
    |   +-- layers.json

To run training process we will use following (config_train.json):

.. code:: bash

    {
    "neural_network": {
        "num_epochs": 6,
        "outfile": "out_classifier.txt",
        "layers": "./layers.json",
        "checkpoints": {
            "load": "./model_checkpoints/model.h5",
            "save": "model_checkpoints_save/",
            "interval": 3
        },
        "data": {
            "input_type": "images",
            "target_type": "labels",
            "input_path":      "./data/train/",
            "test_input_path": "./data/valid/",
            "batch_size": 64,
            "test_batch_size": 128,
            "preload": true,
            "transforms": [
                { "resize": [224, 224] }
            ]
        },
        "optimizer": {
            "type": "Adam",
            "learning_rate": 0.001,
            "lr_scheduler": {
                "type": "Exponential",
                "gamma": 0.5
            }
        },
        "loss": {
            "function": "NLL"
        }
    }
}

Most parameters in the config file are self-explanatory. The most important here are:

 - The neural network description file is specified in ``layers``
 - The input training and test data location is passed in ``input_path`` and ``test_input_path``
 - The optimizer used will be Adam, in which we have used learning rate scheduler which decreases the learning rate exponentially as we train. We have used 0.001 as starting point for our learning rate.
 - The loss function used will be NLL (Negative Log-Likelihood)
 - We will train for 6 epochs using a batch-size of 64 for training and 128 for testing, and write out a model checkpoint file after every 3 epochs.

We will now run training on multi core CPU using a Docker image with `rmldnn`
(see `instructions <https://github.com/rocketmlhq/rmldnn/blob/main/README.md#install>`__ for how to get the image).
From the command line, one should do:

.. code:: bash

   $ sudo docker run --cap-add=SYS_PTRACE -u $(id -u):$(id -g) -v ${PWD}:/home/ubuntu -w /home/ubuntu --rm \
    rocketml/rmldnn:latest mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=8 \
    rmldnn --config=config_test.json

`rmldnn` will configure the run and start the dataset:

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/images/Train_SS.png?raw=true
  :width: 1000

In addition to the information printed on the standard output, `rmldnn` also writes out two log files named after the
``outfile`` parameter in the config file. The file ``out_classifier_train.txt`` reports the loss value and gradient norm
as functions of both time (in secs) as well as the epoch/batch number. The file ``out_classifier_test.txt`` reports loss
and accuracy for running inference on the test dataset (the accuracy for a classification problem is simply the fraction
of correctly labeled data samples).

We can monitor the run by plotting quantities like the training loss and the test accuracy, as shown below.

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/images/test_rpoch_loss.png?raw=true
  :width: 400
  :align: center

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/images/test_rpoch_accuracy.png?raw=true
  :width: 400
  :align: center

Running inference on a pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above run writes out the model trained up to the 6th epoch as ``model_checkpoints_save/model_checkpoint_6.pt``.
This model can be used to run stand-alone inference on a given set of birds species.
For example, assume we want to classify the following 400 random species, which have been
copied under ``test_sample/*.jpg``:

This simple configuration file
(`config_mnist_test.json <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/mnist_classification/config_mnist_test.json>`__)
can be used to run `rmldnn` inference:

.. code:: bash

    {
    "neural_network": {
        "debug": true,
        "outfile": "./predictions.txt",
        "layers": "./layers.json",
        "checkpoints": {
            "load": "./model_checkpoints_save/model_checkpoint_6.pt"
        },
        "data": {
            "input_type": "images",
            "test_input_path": "./test_sample/"
        }
    }
}

To get test_sample folder run following python code inside your main directory:

.. code:: bash

    import os 
    import shutil
    import random
    os.mkdir('test_sample')
    src='./data/test/'
    dest='./test_sample/'
    for directory in os.listdir(src):
        random_file=random.choice(os.listdir(src+directory))
        shutil.copy(src+directory+'/'+random_file,dest)

We can run inference on a multiple CPU by doing:

.. code:: bash

    $ sudo docker run --cap-add=SYS_PTRACE -u $(id -u):$(id -g) -v ${PWD}:/home/ubuntu -w /home/ubuntu --rm \
    rocketml/rmldnn:latest mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=8 \
    rmldnn --config=config_test.json

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/images/Test_SS.png?raw=true
  :width: 1000
  :align: center

The output of classification is a directory named ``predictions/`` containing one small numpy file for each input sample.
Since the model predicts a probability for each sample to be of one out of 400 possible classes, 
those numpy arrays will be of shape :math:`(400,)`. To obtain the actual predictions, one needs to compute
the `argmax` for each array:

.. code:: bash

    import numpy as np
    import os
    for file in sorted(os.listdir('./predictions/')):
        print(np.argmax(np.load('./predictions/' + file)), end=' ')
    
