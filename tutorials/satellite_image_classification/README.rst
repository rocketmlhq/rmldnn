Satellite Image Clasification	
====================================================
	
Introduction
~~~~~~~~~~~~
	
This tutorial explains how to use **rmldnn** to perform transfer learning in order to train a model for the classification of Sentinel-2 satellite images. In this experiment the ResNet-50 CNN models were pretrained on the ImageNet image classification dataset with more than a million images. 

The illustration below shows an overview of the patch-based land use and land cover classiﬁcation process using satellite images. A satellite scans the Earth to acquire images of it. Patches extracted out of these images are used for classiﬁcation.The aim is to automatically provide labels describing the represented physical land type or how the land is used. For this purpose, an image patch is feed into a classiﬁer, in this illustration a neural network, and the classiﬁer outputs the class shownon the image patch.

.. image:: ./figures/patch_extraction.jpg

The resulting classiﬁcation system opens a gate towards a number of Earth observation applications for further advances. 
	
The Dataset
~~~~~~~~~~~
	
We will use the Sentinel-2 satellite images covering 13 spectral bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images. The classes include River, Highway, Forest, Industrial buildings and 6 other classifications of Satellite images. The dataset is split into the ratio of 94:5:1 for train, validation and test respectively. You can get this dataset from here (as a zip file).
	
We'll create a directory named data/ and unzip the archive inside. You should have the following directory structure:

.. code:: bash

    +-- data/
    |   +-- train/
        |   +-- AnnualCrop/
        |   +-- Forest/
        |   +-- ...
        |   +-- SeaLake/
    |   +-- valid/
        |   +-- AnnualCrop/
        |   +-- Forest/
        |   +-- ...
        |   +-- SeaLake/
    |   +-- test/
        |   +-- AnnualCrop/
        |   +-- Forest/
        |   +-- ...
        |   +-- SeaLake/

	
	
The images are multi-channel (RGB) with size 64 X 64, similar to the ones in the figure below.
	
.. image:: ./figures/sample_images.jpg	
	
The Neural Network
~~~~~~~~~~~

In order to perform transfer learning, we'll need to get our base model, which which in our case is RESNET-50 with 2 modifications:

 1. Add a single 10-unit dense layer at the end (with a log-softmax activation). 
 2. Turning all the BatchNormalisation layers as *Trainable*

After that, we'll need to save our prepared model as an HDF5 file and our network architecture as a .json file. The network is depicted below:

.. image:: ./figures/resnet50.jpg

For convenience, the .h5 file is available `here < >` and the network file `layers.json < >` as well. 

Running Training
~~~~~~~~~~~


	
	
