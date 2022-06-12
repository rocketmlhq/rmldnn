Birds Species classification using transfer learning
====================

Introduction
~~~~~~~~~~~~

This tutorial explains how to use 'rmldnn' to perform transfer learning to train a model that classifies birds species images from dataset.(https://www.kaggle.com/datasets/gpiosenka/100-bird-species).

Starting with a brief introduction about Transfer Learning, Transfer learning is a machine learning method in which a model generated for one job is reused as the starting point for a model on a different task. Here we have leveraged pre-trained RESNET50, which is trained on more than a million images from the ImageNet database. RESNET50 is CNN (Convolutional Neural Network) model which is about 50 layers deep. Below Image shows architecture of RESNET50 

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/ResnetArch.png?raw=true
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

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/birds_cover.jpg?raw=true

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


The images are multi-channel(Coloured) with size of 224 X 224, similar to ones in the figure below. 

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/Birds_joined.png?raw=true

The neural network
~~~~~~~~~~~~~~~~~~

Since we'll be doing transfer learning, we'll need to first get our base model, which in our instance is RESNET50, and then add a single 400-unit dense layer at the end (with a log-softmax activation). After that, we'll need to save our prepared model as a Hdf5 file and our network architecture as a .json file so that we can train it with rmldnn. The network is depicted and described below:

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/Birds_Classification/network_arch.png?raw=true
    :height: 500
    :align: center

You can use this notebook to learn about all of the tasks involved in preparing and saving the model and its architecture. We've additionally made .hdf5 and.json availble to download directly from here.

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

To run training process we will use following(config_train.json):

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
 - The optimizer used will be Adam, in which we have used Leraning rate scheduler which decreases the learning rate exponentially as we train. We have used 0.001 as starting point for our learning rate.
 - The loss function used will be NLL (Negative Log-Likelihood)
 - We will train for 6 epochs using a batch-size of 64 for training and 128 for testing, and write out a model checkpoint file after every 3 epochs.

We will now run training on two GPUs using a Singularity image with `rmldnn`
(see `instructions <https://github.com/rocketmlhq/rmldnn/blob/main/README.md#install>`__ for how to get the image).
From the command line, one should do:

.. code:: bash

  $ singularity exec --nv ./rmldnn_image.sif \
    mpirun -np 2 -x CUDA_VISIBLE_DEVICES=0,1 \
    rmldnn --config= ./config_mnist_training.json

`rmldnn` will configure the run and start training on the MNIST dataset:

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/mnist_classification/figures/mnist_run_training.png
  :width: 1000

In addition to the information printed on the standard output, `rmldnn` also writes out two log files named after the
``outfile`` parameter in the config file. The file ``out_mnist_train.txt`` reports the loss value and gradient norm
as functions of both time (in secs) as well as the epoch/batch number. The file ``out_mnist_test.txt`` reports loss
and accuracy for running inference on the test dataset (the accuracy for a classification problem is simply the fraction
of correctly labeled data samples).

We can monitor the run by plotting quantities like the training loss and the test accuracy, as shown below.

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/mnist_classification/figures/mnist_loss.png
  :width: 500
  :align: center

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/mnist_classification/figures/mnist_accuracy.png
  :width: 500
  :align: center

Running inference on a pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above run writes out the model trained up to the 20th epoch as ``mnist_model/model_checkpoint_20.pt``.
This model can be used to run stand-alone inference on a given set of MNIST digits.
For example, assume we want to classify the following 10 random digits, which have been
copied under ``mnist_digits/digit_*.jpg``:

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/mnist_classification/figures/mnist_test_digits.png
  :width: 1000
  :align: center

This simple configuration file
(`config_mnist_test.json <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/mnist_classification/config_mnist_test.json>`__)
can be used to run `rmldnn` inference:

.. code:: bash

    {
        "neural_network": {
            "debug": true,
            "outfile": "./mnist_predictions.txt",
            "layers": "./mnist_keras_net.json",
            "checkpoints": {
                "load": "./mnist_model/model_checkpoint_20.pt"
            },
            "data": {
                "input_type": "images",
                "test_input_path": "./mnist_digits/",
                "grayscale": true
            }
        }
    }

We can run inference on a single CPU by doing:

.. code:: bash

    $ singularity exec rmldnn_image.sif rmldnn --config= ./config_mnist_test.json

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/mnist_classification/figures/mnist_run_testing.png
  :width: 1000
  :align: center

The output of classification is a directory named ``mnist_predictions/`` containing one small numpy file for each input sample.
Since the MNIST model predicts a probability for each sample to be of one out of 10 possible classes, 
those numpy arrays will be of shape :math:`(10,)`. To obtain the actual predictions, one needs to compute
the `argmax` for each array:

.. code:: bash

    import numpy as np
    import os
    for file in sorted(os.listdir('./mnist_predictions/')):
        print(np.argmax(np.load('./mnist_predictions/' + file)), end=' ')
    
    >>> 3 5 1 9 4 7 2 0 6 8 

For this test set, we achieved 100% prediction accuracy with a model trained for only 20 epochs!
This is actually not surprising, given that MNIST is nowadays considered the `hello-world`
of image classification problems.
