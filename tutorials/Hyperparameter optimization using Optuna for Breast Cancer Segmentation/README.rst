Hyperparameter optimization using Optuna for Breast Cancer Segmentation
======================================================================

Introduction
~~~~~~~~~~~~

This tutorial explains how to use **rmldnn** to automate hyperparameter search using optuna and also to use it to perform breast cancer segmentation.

*Hyperparameter Optimization* is the process of choosing the optimal set of hyperparameters to increase the model's performance. It operates by conducting numerous trials within a single training procedure. Every trial entails the full execution of your training application with the values of the selected hyperparameters set within the predetermined bounds. Once this procedure is complete, you will have the set of hyperparameter values that the model requires to perform at its best. Figure below demonstrates the process:
