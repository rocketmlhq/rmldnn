Brain MRI image segmentation
===========================

Introduction
~~~~~~~~~~~~

Image segmentation is a technique that divides a digital image into several groups known as "image segments", which helps to 
identify regions of interest within that image. On digital images, this is usually done by assigning pixels
of different colors to each segment, where each color (pixel value) corresponds to some category of interest. 

Although several image segmentation techniques have been developed over the years (e.g, thresholding, 
histogram-based bundling, k-means clustering, etc), deep-learning has shown to achieve the best accuracy
on a variety of image segmentation problems.

In this tutorial, we will show how to use `rmldnn` to efficiently train an image segmentation model using
a dataset consisting of human brain MRI images. 

The dataset
~~~~~~~~~~~

We will use the `Brain MRI segmentation <https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation>`__
dataset, which contains brain MRI images together with manual FLAIR abnormality segmentation masks, as shown below..  

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/sample.png?raw=true
  :width: 650
  

The data needs to be pre-processed before training. This is done in Keras through code, but
we do it here as an outside step in order to save time when running multiple training experiments. 
We need to:

 - remove non-image files (e.g., ``.mat``) from the dataset
 - partition the images as training (90%) and test (10%) sets
 - organize images in the following directory structure

.. code:: bash

    +-- brain_MRI_image_segmentation/
    |   +-- data/
        |   +-- train_image/
        |   +-- train_mask/
        |   +-- test_image/
        |   +-- test_mask/
        |   +-- sample/
        |   +-- sample_true/

For your convenience, we have already pre-processed the dataset which can be downloaded directly from `here <https://rmldnnstorage.blob.core.windows.net/rmldnn-datasets/brain_MRI.tar.gz>`__

The model
~~~~~~~~~

We will use UNET style neural network backed by RESNET50 for this purpose. The architecture has two paths. The contraction path, also known as the encoder, is the first path and it is used to record the context of the image. The encoder is simply a conventional stack of max pooling and convolutional layers. The second path, also known as the decoder, is a symmetric expanding path that enables exact localisation using transposed convolutions. Because it only has Convolutional layers and no Dense layers, it is an end-to-end fully convolutional network (FCN), allowing it to process images of any size.
Below image shows architecture of RESUNET:

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/unet.png?raw=true
  :width: 600
  :height: 700
  :align: center
  
Fully convolutional neural network RESUNET was created with the goal of achieving great performance with a minimal number of parameters. Over the current UNET design, it is an advancement. RESUNET benefits from the Deep Residual Learning as well as the UNET design. Similar to a U-Net, the RESUNET is made up of an encoding network, a decoding network, and a bridge connecting the two. The U-Net employs two 3 x 3 convolutions, with a ReLU activation function coming after each. In the case of RESUNET, a pre-activated residual block takes the place of these layers. Our RESUNET network is pretrained using ImageNet dataset, which is dataset consisting of hundreds and thousands of images.

You can get the model from `here <https://rmldnnstorage.blob.core.windows.net/rmldnn-models/model_resunet_imagenet.h5>`__.

Training the model
~~~~~~~~~~~~~~~~~~

To train the ResUnet model on our dataset, we will use Adam optimizer with learning rate of 0.0001 along with Exponential learning rate scheduler with gamma of 0.95. To learn more about types of lr scheduler `click here <https://rocketmlhq.github.io/rmldnn/configuration.html#lr-scheduler-sub-section>`__.


However, instead of using a categorical cross-entropy loss function, we will take advantage of `rmldnn`'s implementation
of the Dice loss, which is defined as the complement of the Dice coefficient computed between prediction and target.
First introduced in the context of medical image segmentation
(`paper <https://arxiv.org/abs/1606.04797>`__),
the Dice loss has been shown to perform very well for segmentation tasks in general.

The `rmldnn` configuration file used for training is shown below:

.. code:: bash

  {
      "neural_network": {
          "outfile": "out_segmentation.txt",
          "num_epochs": 20,
          "layers": "./layers_resunet.json",
          "checkpoints": {
              "load": "./model_resunet_imagenet.h5",
              "save": "model_MRI_segmentation/",
              "interval": 10
          },
          "data": {
              "type": "images",
              "input_path":       "./data/train_image/",
              "target_path":      "./data/train_mask/",
              "test_input_path":  "./data/test_image/",
              "test_target_path": "./data/test_mask/",
              "batch_size": 32,
              "test_batch_size": 64,
              "preload": true,
              "target_grayscale": true,
              "target_is_mask": true,
              "transforms": [
                  { "resize": [256, 256] }
              ]
          },
          "optimizer": {
              "type": "adam",
              "learning_rate": 0.0001,
              "lr_scheduler": {
                  "type": "Exponential",
                  "gamma": 0.95,
                  "verbose": true
              }
          },
          "loss": {
              "function": "Dice",
              "source": "sigmoid"
          }
      }
  } 


A few points to notice in the configuration:

 - Since the targets are grayscale images (single-channel), the parameter ``target_grayscale`` is set to `true`,
   otherwise they would be loaded as 3-channel tensors that would not match the target shape 
   expected by the Dice loss function.
 - The variable ``target_is_mask`` is set to `true` so that target pixels are not linearly interpolated 
   when resizing the image.
 - Since we are performing transfer learning, we will load a pre-trained ResUnet model.

We will run training for 20 epochs on 4 NVIDIA V100 GPUs using a Docker image with `rmldnn` 
(see `instructions <https://github.com/rocketmlhq/rmldnn/blob/main/README.md#install>`__ for how to get the image).
From the command line, one should do:

.. code:: bash

   sudo docker run --cap-add=SYS_PTRACE --gpus=all -u $(id -u):$(id -g) -v ${PWD}:/home/ubuntu -w /home/ubuntu --rm \
    rocketml/rmldnn:latest mpirun -np 4 -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
    rmldnn --config=config_train.json

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/train_ss.png?raw=true
  :width: 600
  :align: center

It takes about 4 minutes to train for 20 epochs on 4 GPUs. 
We can monitor the run by plotting quantities like the training loss and the test accuracy, as shown below.

.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/epoch_loss_plot.png?raw=true
  :width: 400
  :align: center
  
.. image:: https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/epoch_acc_plot.png?raw=true
  :width: 400
  :align: center
  
The test accuracy, reported in the file ``out_segmentation_test.txt``, shows that we have reached
an accuracy of ~87% on the test dataset (as measured by the Dice coefficient averaged across all classes).


Running inference on a pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's now use the model saved after the 20th epoch to run inference on a few samples and visualize the results.
We have made a copy of about 4 test images under ``./data/sample``, which you can use to run inference on. Use following configuration file to run inference:

.. code:: bash

  {
      "neural_network": {
          "debug": true,
          "layers": "./layers_resunet.json",
          "checkpoints": {
              "load": "./model_MRI_segmentation/model_checkpoint_20.pt"
          },
          "data": {
              "type": "images",
              "test_input_path":  "./data/sample/",
              "test_batch_size": 16,
              "transforms": [
                  { "resize": [256, 256] }
              ]
          }
      }
  }

The setting ``debug = true`` instructs `rmldnn` to save the predictions as ``numpy`` files under ``./debug/``.

We can run inference on the test images by doing:

.. code:: bash

    sudo docker run --gpus=all -u $(id -u):$(id -g) -v ${PWD}:/home/ubuntu -w /home/ubuntu --rm \
      rocketml/rmldnn:latest rmldnn --config=config_test.json 

Finally, we can visualize the predictions, for example, by loading the `numpy` files and showing the images
with `matplotlib`.

.. code:: bash

    import numpy as np
    import matplotlib.pyplot as plt

    pred = np.load('./debug/output_1_0.npy').round()
    plt.imshow(pred[0,:,:],cmap="gray")
    plt.show()

Doing this for a few samples, we obtain the segmentation predictions below.
Results are pretty good for a model trained for only 10 minutes! 

==================== ==================== ====================
**Inputs**           **Predictions**      **Ground-truths**
-------------------- -------------------- --------------------
|input_1|            |inference_1|        |truth_1|
-------------------- -------------------- --------------------
|input_2|            |inference_2|        |truth_2|
-------------------- -------------------- --------------------
|input_3|            |inference_3|        |truth_3|
-------------------- -------------------- --------------------
|input_4|            |inference_4|        |truth_4|
==================== ==================== ====================

.. |input_1|      image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/input_1.png?raw=true
    :width: 300
.. |input_2|      image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/input_2.png?raw=true
    :width: 300
.. |input_3|      image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/input_3.png?raw=true
    :width: 300
.. |input_4|      image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/input_4.png?raw=true
    :width: 300
.. |inference_1|  image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/pred_1.png?raw=true
    :width: 300
.. |inference_2|  image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/pred_2.png?raw=true
    :width: 300
.. |inference_3|  image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/pred_3.png?raw=true
    :width: 300
.. |inference_4|  image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/pred_4.png?raw=true
    :width: 300
.. |truth_1|      image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/true_1.png?raw=true
    :width: 300
.. |truth_2|      image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/true_2.png?raw=true
    :width: 300
.. |truth_3|      image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/true_3.png?raw=true
    :width: 300
.. |truth_4|      image::  https://github.com/yashjain-99/rmldnn/blob/main/tutorials/brain_MRI_image_segmentation/figures/true_4.png?raw=true
    :width: 300
   
