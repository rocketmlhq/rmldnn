rmldnn smoke-tests
==================

Installing Singularity CE
~~~~~~~~~~~~~~~~~~~~~~~~~

Build Singularity from sources by following the directions in this document:

https://sylabs.io/guides/3.9/user-guide/quick_start.html#quick-installation-steps


Downloading the Singularity image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download the rmldnn singularity image from:

TBD

and point the following environment variable to its location:

.. code:: bash

  export RMLDNN_IMAGE=/path/to/rmldnn_image.sif

Running rmldnn smoke tests
~~~~~~~~~~~~~~~~~~~~~~~~~~

Enter the smoke-test directory:

.. code:: python

    cd smoke_tests/

1. Running single-process tests on CPU
--------------------------------------

.. code:: bash

  $ singularity exec ${RMLDNN_IMAGE} rmldnn --config= ./config_rmldnn_test.json 

Success criteria:
 - The output should be similar to the log file in ``baseline_rmldnn_test.txt``
 - Check that the files ``out_dnn_mnist_sgd_train.txt`` and ``out_dnn_mnist_sgd_test.txt`` have been created and have data.


2. Running multi-process (parallel) tests on CPU
------------------------------------------------

.. code:: bash

  $ rm -f out_*
  $ singularity exec ${RMLDNN_IMAGE} mpirun -np 2 -bind-to none -x OMP_NUM_THREADS=1 rmldnn --config= ./config_rmldnn_test.json 
 
Success criteria:
 - The output should contain the message: ``RocketML : 2 MPI processes``
 - Check that the files ``out_dnn_mnist_sgd_train.txt`` and ``out_dnn_mnist_sgd_test.txt`` have been created and have data.


3. Running single-GPU tests
---------------------------

Run the ``nvidia-smi`` utility to check that the node has available GPUs and CUDA/CuDNN have been installed.
If the command is not found, or cannot connect to device, the following tests cannot be performed.

.. code:: bash

  $ rm -f out_*
  $ singularity exec --nv ${RMLDNN_IMAGE} rmldnn --config= ./config_rmldnn_test.json

Success criteria:
 - The output should contain the message: ``CUDA available! Will train on GPU``
 - Check that the files ``out_dnn_mnist_sgd_train.txt`` and ``out_dnn_mnist_sgd_test.txt`` have been created and have data.


4. Running multi-GPU tests
--------------------------

Run the ``nvidia-smi`` utility to check that the node has multiple GPUs and CUDA/CuDNN have been installed.
If the command is not found, or cannot connect to device, the following GPU tests cannot be performed.
Also, if only one device is available (i.e., single-GPU node), the following tests cannot be performed.

.. code:: bash

  $ rm -f out_*
  $ singularity exec --nv ${RMLDNN_IMAGE} mpirun -np 2 -x CUDA_VISIBLE_DEVICES=0,1 rmldnn --config= ./config_rmldnn_test.json 

Success criteria:
 - The output should contain the message: ``CUDA:0`` and ``CUDA:1``
 - Check that the files ``out_dnn_mnist_sgd_train.txt`` and ``out_dnn_mnist_sgd_test.txt`` have been created and have data.
