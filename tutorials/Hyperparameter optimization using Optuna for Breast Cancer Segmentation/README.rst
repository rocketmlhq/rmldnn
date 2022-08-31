Hyperparameter optimization using Optuna for Breast Cancer Segmentation
======================================================================

Introduction
~~~~~~~~~~~~

This tutorial explains how to use **rmldnn** to automate hyperparameter search using optuna and also to use it to perform breast cancer segmentation.

*Hyperparameter Optimization* is the process of choosing the optimal set of hyperparameters to increase the model's performance. It operates by conducting numerous trials within a single training procedure. Every trial entails the full execution of your training application with the values of the selected hyperparameters set within the predetermined bounds. Once this procedure is complete, you will have the set of hyperparameter values that the model requires to perform at its best. Figure below demonstrates the process:

.. image:: ./figures/flowchart.png?raw=true
    :width: 550
    :height: 300
    :align: center
   
What is Optuna?
~~~~~~~~~~~~~~~

Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It automatically finds optimal hyperparameter values by making use of different samplers such as grid search, random, bayesian, and evolutionary algorithms.

We use the terms study and trial as follows:
 - Study: optimization based on an objective function
 - Trial: a single execution of the objective function
 
Features:
 - Eager dynamic search spaces
 - Efficient sampling and pruning algorithms
 - Easy integration
 - Good visualizations
 - Distributed optimization
 
Click `here <https://optuna.org/>`__ to know more about optuna.

What is Typer?
~~~~~~~~~~~~~~

Typer is a library for building Command Line Interface (CLI) applications based on Python’s type hints. It is created by Sebastián Ramírez, the author of FastAPI. 

Features:
 - Intuitive to write: Great editor support. Completion everywhere. Less time debugging. Designed to be easy to use and learn. Less time reading docs.
 - Easy to use: It's easy to use for the final users. Automatic help, and automatic completion for all shells.
 - Short: Minimize code duplication. Multiple features from each parameter declaration. Fewer bugs.
 - Start simple: The simplest example adds only 2 lines of code to your app: 1 import, 1 function call.
 - Grow large: Grow in complexity as much as you want, create arbitrarily complex trees of commands and groups of subcommands, with options and arguments.
 
Click `here <https://typer.tiangolo.com/>`__ to know more about typer.

The Dataset
~~~~~~~~~~~

To explain the working of our script we have choosen `Breast Ultrasound Images Dataset <https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset>`__ to perform Breast Cancer Segmentation on ultrasound images. This kaggle dataset consists of 780 images with an average image size of 500*500 pixels collected from around 600 female patients. For convenience, we have already pre-processed the dataset, which can be downloaded directly from `here <https://rmldnnstorage.blob.core.windows.net/rmldnn-datasets/brain_MRI.tar.gz>`__. 
When saving the dataset kindly make sure that you maintain following directory structure:

.. code:: bash

    +-- breast_cancer_segmentation/
    |   +-- data/
        |   +-- train/
            |   +-- inputs/
            |   +-- targets/
        |   +-- test/
            |   +-- inputs/
            |   +-- targets/
            
The model
~~~~~~~~~

Model that we will be using is an UNET styled network trained on RESNET. To know more about this model and it's working kindly refer to our tutorial on `Brain MRI Segmentation <https://github.com/yashjain-99/rmldnn/tree/main/tutorials/brain_MRI_image_segmentation>`__.
The pre-trained model can be downloaded from `here <https://rmldnnstorage.blob.core.windows.net/rmldnn-models/model_resunet_imagenet.h5>`__..

Steps to Automate the task of Hyper-Parameter optimization using RMLDNN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Perform the following steps:
 #. Download the python scripts provided here and save it in the same directory as your data folder.
 #. Run below mentioned command to install required libraries
 
 .. code:: bash
    
    pip install shutil typer optuna
    
 #. Now open the terminal and navigate to your directory, after that type in **Python RML_typer.py --help** which will print out available options. Below Lines will describe each option available in detail and we will also construct the command for Breast Cancer segmentation on the go with that.
 #. First argument that it requires is number of trials you want for optuna to run for. This argument is required and cannot be skipped. You can add in numrical values here. In our case we are going to go for 50 trials so we will be adding **--num-trials 50** or also you could use -nt 50.
 #. Second argument that it requires is number of epochs you want for optuna to run for per trial. This argument is required and cannot be skipped. You can add in numrical values here. In our case we are going to go for 50 epochs so we will be adding **--num-epochs 50** or also you could use -ne 50.
 #. Third and fourth arguments are optional which allows you to choose between docker or singularity container to run RMLDNN. You could choose any and provide in respective image required for that container. For default it is set to docker with rocketml/rmldnn:latest image. In our case we will going with default docker container so will be adding in **-docker** to our command.
 #. Fifth argument is used when you want to use gpu's to speed up training process. To do so add in --gpu or just skip it if you don't want to use. Since we will be using a gpu system so will be adding **--gpu** to our command.
 #. Sixth argument is used when you have multiple cores available in your system and want to utilize them. To do so just add in --multi-core to your command and then later while running, it will prompt you to enter in number of cores you want to use. Since we will be training on single core GPU system so we will be skipping this part here.
 #. Seventh argument is required and asks you to enter optimizers you want to test your model with. To enter optimizers make sure they are comma seperated. In our case we are going to go for adam, rmsprop and sgd so we will be adding **--optimizers adam,rmsprop,sgd** or -o adam,rmsprop,sgd to our command.
 #. Eight argument is required and asks you to enter loss functions you want to test your model with. To enter loss functions make sure they are comma seperated. This argument is also required and can not be skipped. In our case we are going to go for bce and dice so we will be adding **--loss bce,dice** or -l bce,dice.
 #. Ninth argument ask you to enter any learning rate of your choice. This is an optional argument with default learning rate of 0.001 but you can add in any value that you desire for example --learning-rate 0.0001 or -lr 0.0001. In our case we will be skipping this option.
 #. Tenth argument asks you enter file name which contains model architecture, this also an optional argument with default value of layers.json. In our case we will be adding **--layers layers_resunet.json** to our command.
 #. Eleventh argument is used when you want to use Learning rate scheduler while training. This is an optiional argument and can be skipped. In our case we will be adding **--lr-scheduler** to our command. This will later prompt us with start and end value of learning rate scheduler as well as gamma value for the same. The values that we will be entering are 1e-4, 1e-1 and 0.95 respectively. Note: As of now we have only allowed Exponential learning rate scheduler which is also set as default value for the same.
 #. Twelfth argument is used when you want to implement transfer learning while training. This is an optiional argument and can be skipped. In our case we will be adding **--transfer-learning** to our command. This will later prompt us to enter file name for the same which in our case will be model_resunet_imagenet.h5, do make sure this file is in the same location as the script or else enter the complete path for that file.
 
Adding up all these leads to following final command

.. code:: bash

    Python RML_typer.py --num-trials 50 --num-epochs 50 -docker --gpu --optimizers adam,rmsprop,sgd --loss bce,dice --layers layers_resunet.json --lr-scheduler --transfer-learning 
    
On succesfully running, above command will start the process for given number of trials. On finishing the last trial it will save a log file with record of accuracies found in each trial along with other parameters. As well as it will save best performing model inside a folder named best_model. This model can then later be used for running infernce. 

Running inference on pre-trained model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For running inference using best performing model we will need following configuration file(`config_test.json <>`__):

.. code:: bash

  {
      "neural_network": {
          "layers": "./layers_resunet.json",
          "checkpoints": {
              "load": "./best_model/model_checkpoint_50.pt"
          },
          "data": {
              "type": "images",
              "test_input_path":  "./data/sample/",
              "test_output_path": "./predictions/",
              "test_batch_size": 16,
              "transforms": [
                  { "resize": [256, 256] }
              ]
          }
      }
  }

``Note: Kindly change model file name as what is there inside best_model directory.``

This will save the predictions as an ``HDF5`` file under ``./predictions/``.

We can run inference on the test images by doing:

.. code:: bash

    sudo docker run --gpus=all -u $(id -u):$(id -g) -v ${PWD}:/home/ubuntu -w /home/ubuntu --rm \
      rocketml/rmldnn:latest rmldnn --config=config_test.json 
     
Finally, we can visualize the predictions by loading each dataset in the `HDF5` file
and showing the images with `matplotlib`:

.. code:: bash

  import numpy as np
  import h5py as h5
  import matplotlib.pyplot as plt

  pred = h5.File('predictions/output_1.h5', 'r')
  for dataset in pred:
    plt.imshow(pred[dataset][0,:,:].round(), cmap="gray")
    plt.show()
   
Doing this for a few samples, we obtain the segmentation predictions below.

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

.. |input_1|      image::  ./figures/input_1.png?raw=true
    :width: 300
    :height: 300
.. |input_2|      image::  ./figures/input_2.png?raw=true
    :width: 300
    :height: 300
.. |input_3|      image::  ./figures/input_3.png?raw=true
    :width: 300
    :height: 300
.. |input_4|      image::  ./figures/input_4.png?raw=true
    :width: 300
    :height: 300
.. |inference_1|  image::  ./figures/pred_1.png?raw=true
    :width: 300
    :height: 300
.. |inference_2|  image::  ./figures/pred_2.png?raw=true
    :width: 300
    :height: 300
.. |inference_3|  image::  ./figures/pred_3.png?raw=true
    :width: 300
    :height: 300
.. |inference_4|  image::  ./figures/pred_4.png?raw=true
    :width: 300
    :height: 300
.. |truth_1|      image::  ./figures/truth_1.png?raw=true
    :width: 300
    :height: 300
.. |truth_2|      image::  ./figures/truth_2.png?raw=true
    :width: 300
    :height: 300
.. |truth_3|      image::  ./figures/truth_3.png?raw=true
    :width: 300
    :height: 300
.. |truth_4|      image::  ./figures/truth_4.png?raw=true
    :width: 300
    :height: 300
   
