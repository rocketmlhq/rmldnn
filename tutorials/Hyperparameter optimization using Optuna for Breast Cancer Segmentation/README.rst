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
   
What is Optuna
~~~~~~~~~~~~~~

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
 
