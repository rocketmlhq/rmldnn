Birds Species classification using transfer learning
====================================================

Introduction
~~~~~~~~~~~~

This tutorial explains how to use **rmldnn** to perform transfer learning in order to train a model that classifies bird species images.

*Transfer learning* is a machine learning method in which a model trained with a given dataset is reused as starting point for a different task. Here we have leveraged a pre-trained RESNET50 model, which was trained on more than a million images from the ImageNet dataset. RESNET50 is a CNN (Convolutional Neural Network) model which is about 50 layers deep. The image below shows the architecture of RESNET50:

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/birds_image_classification/images/ResnetArch.png?raw=true
    :width: 750
    :align: center
  
The above tasks will exemplify how to use rmldnn to:

 - perform `transfer learning` by reusing model weights in a different model;
 - train a convolutional neural network;
 - use a learning rate scheduler to decay the learning rate in order to speed up convergence.


The dataset
~~~~~~~~~~~

We will use the *Kaggle Birds 400* dataset, which contains 62K images of 400 species of birds, from which 58388 are training images, 2000 test images (5 images per species) and 2000 are validation images (5 images per species). The classes include: ABBOTTS BABBLER, ABBOTTS BOOBY, ABYSSINIAN GROUND HORNBILL and 397 other bird species. You can get this dataset from  `here <https://www.kaggle.com/datasets/gpiosenka/100-bird-species>`__ (as a zip file).

**IMPORTANT:** There is an error in the training set from Kaggle: the directory "BLACK & YELLOW BROADBILL" contains an extra space that is not present in the validation or testing sets. Please rename this directory in the training set by removing the extra space before proceeding.

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/birds_image_classification/images/birds_cover.jpg?raw=true


We'll create a directory named ``data/`` and unzip the archive inside. You should have the following directory structure:

.. code:: bash

    +-- data/
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


The images are multi-channel (colored) with size 224 X 224, similar to the ones in the figure below. 

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/birds_image_classification/images/Birds_joined.png?raw=true

The neural network
~~~~~~~~~~~~~~~~~~

Since we'll be doing transfer learning, we'll need to first get our base model, which in our case is RESNET50, and then add a single 400-unit dense layer at the end (with a log-softmax activation). After that, we'll need to save our prepared model as an HDF5 file and our network architecture as a .json file. The network is depicted below:

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/birds_image_classification/images/network_arch.png?raw=true
    :height: 500
    :align: center

For reference, the above steps (fetching and exporting the model) can be accomplished with the script below. However, for convenience, the .h5 is available from  `here <https://rmldnnstorage.blob.core.windows.net/rmldnn-models/model_resnet50_imagenet.h5>`__, and the network file
`layers.json <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/birds_image_classification/layers.json>`__
is provided with this tutorial.

.. code:: python

    import json
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model
    
    pretrained_model = ResNet50(
        input_shape=(224,224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    pretrained_model.trainable = False
    
    # Add dense layer with 400 units and log_softmax activation after base model
    inputs = pretrained_model.input
    outputs = Dense(400, activation='log_softmax')(pretrained_model.output)
    model = Model(inputs=inputs, outputs=outputs)
    
    # Export model as HDF5
    model.save("model_resnet50_imagenet.h5")
    
    # Save network architecture in json format
    net = json.dumps(json.loads(model.to_json()), indent=4)
    with open("layers.json",'w') as f:
        f.write(net)
        

Running training
~~~~~~~~~~~~~~~~

**rmldnn** is a code-free, high-performance tool for distributed deep-learning, and the entire flow can be defined
in a single configuration file. We will assume the following directory structure inside the main folder:

.. code:: bash

    +-- birds_image_classification/
    |   +-- data/
        |   +-- train/
        |   +-- test/
        |   +-- valid/
    |   +-- model_resnet50_imagenet.h5
    |   +-- layers.json

To run training, we will use the following configuration file
(`config_train.json <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/birds_image_classification/config_train.json>`__):

.. code:: json

    {
    "neural_network": {
        "num_epochs": 6,
        "outfile": "out_classifier.txt",
        "layers": "./layers.json",
        "checkpoints": {
            "load": "./model_resnet50_imagenet.h5",
            "save": "model_checkpoints_save/",
            "interval": 2
        },
        "data": {
            "input_type":  "images",
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

 - The number of epochs is set to 6, since test accuracy was found to peak at around 5 to 6 epochs during training.
 - The neural network description file is specified in ``layers``
 - The input training and test data locations are passed in ``input_path`` and ``test_input_path``
 - The optimizer used will be Adam, with a learning rate scheduler which lowers the learning rate exponentially as we train. We have used 0.001 as the initial learning rate.
 - The loss function used will be NLL (Negative Log-Likelihood)
 - We will use a batch-size of 64 for training and 128 for testing, and write out a model checkpoint file after every 2 epochs.

We will run training on a multi-core CPU node using a Docker image with `rmldnn`
(see `instructions <https://github.com/rocketmlhq/rmldnn/blob/main/README.md#install>`__ for how to get the image).
The following command will run training in parallel by spawning 4 processes, each using 8 threads:

.. code:: bash

   $ sudo docker run --cap-add=SYS_PTRACE -u $(id -u):$(id -g) -v ${PWD}:/home/ubuntu -w /home/ubuntu --rm \
     rocketml/rmldnn:latest mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=8 \
     rmldnn --config=config_train.json

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/birds_image_classification/images/train_SS.png?raw=true
  :width: 800

In addition to the information printed on the standard output, `rmldnn` also writes out two log files named after the
``outfile`` parameter in the config file. The file ``out_classifier_train.txt`` reports the loss value and gradient norm
as functions of both time (in secs) as well as the epoch/batch number. The file ``out_classifier_test.txt`` reports loss
and accuracy for running inference on the test dataset (the accuracy for a classification problem is simply the fraction
of correctly labeled data samples).

We can monitor the run by plotting quantities like the training loss and the test accuracy, as shown below.

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/birds_image_classification/images/test_rpoch_loss.png?raw=true
  :width: 400
  :align: center

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/birds_image_classification/images/test_rpoch_accuracy.png?raw=true
  :width: 400
  :align: center

Running inference on a pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above run writes out the model trained up to the 6th epoch as ``model_checkpoints_save/model_checkpoint_6.pt``.
This model can be used to run stand-alone inference on a given set of birds images.
For example, the below script (
`test_sample.py <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/birds_image_classification/test_sample.py>`__)
will copy one random image from each bird species (to a total of 400 imges) into a new ``test_samples/`` directory:

.. code:: python

    import os 
    import shutil
    import random

    src='./data/test/'
    dest='./test_samples/'
    
    os.mkdir(dest)
    
    for directory in os.listdir(src):
        random_file = random.choice(os.listdir(src + directory))
        shutil.copy(src + directory + '/' + random_file, dest)
        os.rename(dest + random_file, dest + directory + random_file)

The following configuration file
(`config_test.json <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/birds_image_classification/config_test.json>`__)
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
            "test_input_path": "./test_samples/",
            "transforms": [
                { "resize": [224, 224] }
            ]
        }
    }


We will run inference in parallel using 4 processes (8 threads each) on a multi-core CPU node:

.. code:: bash

    $ sudo docker run --cap-add=SYS_PTRACE -u $(id -u):$(id -g) -v ${PWD}:/home/ubuntu -w /home/ubuntu --rm \
      rocketml/rmldnn:latest mpirun -np 4 --bind-to none -x OMP_NUM_THREADS=8 \
      rmldnn --config=config_test.json

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/birds_image_classification/images/Test_SS.png?raw=true
  :width: 800
  :align: center

The output of classification is a directory named ``predictions/`` containing one numpy file for each input sample.
Since the model predicts a probability for each sample to be of one out of 400 possible classes, 
those numpy arrays will be of shape :math:`(400,)`. To obtain the actual predicted classes, one needs to take 
the `argmax` of each array. This is done in the below script (available as 
`print_predictions.py <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/birds_image_classification/print_predictions.py>`__),
which also computes the total accuracy:

.. code:: python

    import numpy as np
    import os

    right = 0
    size  = 400

    for i in range(size):
        x = np.argmax(np.load('./predictions/output_1_' + str(i) +'.npy'))
        print(x, end=' ')
        if (x == i):
            right += 1

    print("\n\nAccuracy is " + str(100 * right / size) +'%')

Since our test dataset contains one image from each bird species in order, the above script should print a sequence from 0 to 399, 
if all predictions are correct. In reality, we get an accuracy of about 95%, which is great for a classification problem 
with 400 classes trained for only 6 epochs, showing the power of the transfer learning method.

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/birds_image_classification/images/Test_inference_SS.png?raw=true
  :width: 800
  :align: center
  
    
