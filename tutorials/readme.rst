Birds Species classification using transfer learning
====================

Introduction
~~~~~~~~~~~~

This tutorial explains how to use 'rmldnn' to perform transfer learning to train a model that classifies birds species images from dataset.(https://www.kaggle.com/datasets/gpiosenka/100-bird-species).

Starting with a brief introduction about Transfer Learning, Transfer learning is a machine learning method in which a model generated for one job is reused as the starting point for a model on a different task. Here we have leveraged pre-trained RESNET50, which is trained on more than a million images from the ImageNet database. RESTNET50 is CNN (Convolutional Neural Network) model which is about 50 layers deep. Below Image shows architecture of RESNET50 

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/ResnetArch.png?raw=true
    :width: 750
    :align: center
  
The above tasks will exemplify how to use `rmldnn` to:

 - train a convolutional auto-encoder neural network;
 - perform `transfer learning` by reusing model weights in a different model;
 - use the `random patch` feature of the image dataset loader to generate input/target pairs for inpainting.


The dataset
~~~~~~~~~~~

The MNIST dataset is commonly downloaded as a set of binary files that must be unpacked
into the actual image files (as jpeg, png, etc) or, alternatively, directly loaded as binaries 
using one of the available MNIST dataset loaders out there 
(e.g., from `torchvision <https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST>`__).
We will use image files as input data in order to demonstrate the generic `image` data loader in `rmldnn`.
For that, MNIST images can be downloaded as JPEG files from `here <https://rmldnnstorage.blob.core.windows.net/rmldnn-datasets/mnist.tgz>`__ (as a single ZIP file).
We'll assume that one has unzipped the images into the following directory structure:

.. code:: bash

    +-- mnist/
    |   +-- training/
        |   +-- 0/
        |   +-- 1/
        |   +-- ...
        |   +-- 9/
    |   +-- testing/
        |   +-- 0/
        |   +-- 1/
        |   +-- ...
        |   +-- 9/


There are a total of 60000 images for training (across all 10 classes), and 10000 for testing (or evaluation).
They are all single-channel (grayscale) images of size 28 x 28, similar to the ones in the figure below.

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/mnist_classification/figures/mnist.png

The neural network
~~~~~~~~~~~~~~~~~~

Since MNIST is a very simple dataset to train with, we will use a small (shallow) neural network
consisting of two convolutional layers and a single dense layer at the end (with a log-softmax activation), 
as shown in the figure below. This network can be easily coded (e.g., using 
`Keras <https://keras.io/>`__) and is available here in the file
`mnist_keras_net.json <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/mnist_classification/mnist_keras_net.json>`__.

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/mnist_classification/figures/mnist_net.png

Running training
~~~~~~~~~~~~~~~~

`rmldnn` is a code-free, high-performance tool for distributed deep-learning, and the entire flow can be defined
in a single configuration file. To run MNIST training, we will use the following
(`config_mnist_training.json <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/mnist_classification/config_mnist_training.json>`__):

.. code:: bash

    {
        "neural_network": {
            "outfile": "out_mnist.txt",
            "num_epochs": 20,
            "layers": "./mnist_keras_net.json",
            "checkpoints": {
                "save": "./mnist_model/",
                "interval": 20
            },
            "data": {
                "input_type": "images",
                "target_type": "labels",
                "input_path":      "./mnist/training/",
                "test_input_path": "./mnist/testing/",
                "batch_size": 128,
                "grayscale": true,
                "preload": true
            },
            "optimizer": {
                "type": "Adam",
                "learning_rate": 1e-4
            },
            "loss": {
                "function": "NLL"
            }
        }
    }

Most parameters in the config file are self-explanatory. The most important here are:

 - The neural network description file is specified in ``layers``
 - The input training and test data location is passed in ``input_path`` and ``test_input_path``
 - The optimizer used will be Adam, with a learning rate of 1e-4
 - The loss function used will be NLL (Negative Log-Likelihood)
 - We will train for 20 epochs using a batch-size of 128, and write out a model checkpoint file at the end of the 20th epoch.

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
