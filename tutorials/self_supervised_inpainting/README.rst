Self-supervised image classification using context-encoders and inpainting
==========================================================================

Introduction
~~~~~~~~~~~~

In this tutorial, we will explore a self-supervised technique that leverages transfer learning
by (i) training an auto-encoder neural network on a feature extraction task using an unlabeled dataset, and
(ii) transfering (most of) the auto-encoder weights to another model and training it on an 
image classification task with a sparsely-labeled dataset.

The unsupervised feature extraction task is performed using a technique called
**inpainting** (`paper <https://arxiv.org/pdf/1604.07379.pdf>`__),
a context-based pixel prediction strategy where a convolutional auto-encoder is trained
to generate missing regions of the input images. In the process, this `context-encoder`
model learns a representation that captures the semantics of the visual structures
present in the types of images it is trained on.
The weights from the encoder portion of this model are then transferred to a 
classifier model, which is further trained on a labeled dataset. The entire scheme is depicted below.

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/figures/inpainting.png
  :width: 750
  :align: center

The above tasks will exemplify how to use `rmldnn` to:

 - train a convolutional auto-encoder neural network;
 - perform `transfer learning` by reusing model weights in a different model;
 - use the `random patch` feature of the image dataset loader to generate input/target pairs for inpainting.

The dataset
~~~~~~~~~~~

We will use Kaggle `Natual Images` dataset, which contains JPEG 6,899 images from 8 distinct classes compiled 
from various sources. The classes include airplane, car, cat, dog, flower, fruit, motorbike and person.
It can be downloaded from
`here <https://www.kaggle.com/datasets/prasunroy/natural-images>`__.

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/figures/kaggle.png
  :width: 750

We will partition the data by allocating ~10% of the images from each class as test samples. 
The following directory structure will be created:

.. code:: bash

    +-- natural_images/
    |   +-- training/
        |   +-- airplane/
        |   +-- car/
        |   +-- cat/
        |   +-- dog/
        |   +-- flower/
        |   +-- fruit/
        |   +-- motorbike/
        |   +-- person/
    |   +-- testing/
        |   +-- airplane/
        |   +-- car/
        |   +-- cat/
        |   +-- dog/
        |   +-- flower/
        |   +-- fruit/
        |   +-- motorbike/
        |   +-- person/

Training the context-encoder model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the feature extraction phase, we will train a neural network composed of the encoder portion of the
Resnet-50 network (until the ``bn5c_branch2c`` layer), followed by 5 transposed convolution layers which act as 
up-sampling steps to bring the tensor size back to the original input size, as shown in the figure below.
This neural network is described in the file
`network_resnet50_feature_extract.json <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/network_resnet50_feature_extract.json>`__.

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/figures/resnet50_feature_extract.png
  :width: 750

To implement the inpainting strategy, random patches must be cut out of the input images and used as targets. 
Although this can be done as a data pre-processing step, `rmldnn` provides an in-memory patch generator that 
not only saves time and disk space, but also results in larger data variety by producing different random patches
at each epoch. It can be parametrized with the linear ``size`` and ``number`` of patches. 
For example, the following configuration results in the input/target pair shown below,
with 100 random (possibly overlapping) patches of size 10 x 10:

.. code:: bash

    "transforms": [
        { "random_patches": { "number": 100, "size": 10 } }
    ]

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/figures/random_patches.png
  :width: 500
  :align: center

The following config file
(`config_inpaint_feature_extraction.json <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/config_inpaint_feature_extraction.json>`__)
will be used to configure the feature extraction run:

.. code:: bash

    {
        "neural_network": {
            "num_epochs": 100,
            "outfile": "out_inpaint_feature_extraction.txt",
            "layers": "./network_resnet50_feature_extract.json",
            "checkpoints": {
                "save": "./model_checkpoints/",
                "interval": 10
            },
            "data": {
                "type": "images",
                "input_path":  "./natural_images/training/",
                "target_path": "./natural_images/training/",
                "batch_size": 128,
                "preload": true,
                "transforms": [
                    { "resize": [128, 128] },
                    { "random_patches": {"number": 16, "size": 16} }
                ]
            },
            "optimizer": {
                "type": "Adam",
                "learning_rate": 0.001
            },
            "loss": {
                "function": "BCE"
            }
        }
    }

We will run training on 4 GPUs using a Singularity image with `rmldnn` 
(see `instructions <https://github.com/rocketmlhq/rmldnn/blob/main/README.md#install>`__ for how to get the image).
From the command line, one should do:

.. code:: bash

  $ singularity exec --nv ./rmldnn_image.sif \
    mpirun -np 4 -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
    rmldnn --config= ./config_inpaint_feature_extraction.json

`rmldnn` will configure the run and start training the model. We will tain for 100 epochs,
and can monitor the progress by looking at the time decay of the loss value,
which is reported in the log file ``out_inpaint_feature_extraction_train.txt``:

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/figures/loss_feat_extract.png
  :width: 500
  :align: center


Training the classifier model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to put together the classifier neural network, which we construct by
taking the encoder-only portion of Resnet-50 (up until the ``bn5c_branch2c`` layer),
and add a Dense layer with a softmax activation function at the end. This network is 
depicted below and described in the file
`network_resnet50_classifier.json <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/network_resnet50_classifier.json>`__.

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/figures/resnet50_classifier.png
  :width: 500
  :align: center

Because the encoder layers have exactly the same names in both networks, 
their weights will be transfered from the feature extraction network into the classifier
network when we load the model checkpoint in the next run. Then, we can freeze the weights
of certain layers by setting ``trainable = false``, in which case only the remaining (unfrozen) layers
would be further trained. The more unfrozen layers we have, the better the final accuracy will be, but
the longer it will take to train the classifier. The Dense layer must be trained from scratch, of course.

The following file,
`config_inpaint_classifier.json <https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/config_inpaint_classifier.json>`__,
will be used to train the classifier:

.. code:: bash

    {
        "neural_network": {
            "num_epochs": 100,
            "outfile": "out_inpaint_classifier.txt",
            "layers": "./network_resnet50_classifier.json",
            "checkpoints": {
                "load": "./model_checkpoints/model_checkpoint_100.pt"
            },
            "data": {
                "input_type": "images",
                "target_type": "labels",
                "input_path":      "./natural_images/training/",
                "test_input_path": "./natural_images/testing",
                "batch_size": 128,
                "test_batch_size": 1024,
                "preload": true,
                "transforms": [
                    { "resize": [128, 128] }
                ]
            },
            "optimizer": {
                "type": "Adam",
                "learning_rate": 0.001
            },
            "loss": {
                "function": "NLL"
            }
        }
    }

We will again train on 4 GPUs for 100 epochs using `rmldnn` on a Singularity image:

.. code:: bash

  $ singularity exec --nv ./rmldnn_image.sif \
    mpirun -np 4 -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
    rmldnn --config= ./config_inpaint_classification.json

Notice how `rmldnn` warns about not finding the weights and biases for the Dense layer when loading
the model checkpoint from the feature extraction run. This is expected, since this layer is
new in the classifier network, and precisely what we want to train.

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/figures/run_classifier.png
  :width: 800
  :align: center

We monitor the NLL loss value for the classification run
(reported in ``out_inpaint_classifier_train.txt``)
and make sure it achieves a steady state before 100 epochs:

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/figures/loss_classification.png
  :width: 500
  :align: center

Finally, we look at the accuracy for the test data classification, 
computed as the fraction of correctly labeled samples
(reported in ``out_inpaint_classifier_test.txt``):

.. image:: https://github.com/rocketmlhq/rmldnn/blob/main/tutorials/self_supervised_inpainting/figures/accuracy_classification.png
  :width: 500
  :align: center
