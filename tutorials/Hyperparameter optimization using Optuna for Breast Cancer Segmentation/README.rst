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

To explain the working of our script we have choosen `Breast Ultrasound Images Dataset <https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset>`__ to perform Breast Cancer Segmentation on ultrasound images. This kaggle dataset consists of 780 images with an average image size of 500*500 pixels collected from around 600 female patients.  
