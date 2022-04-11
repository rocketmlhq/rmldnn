*rmltf* installation
====================

Dependencies
------------

:code:`rmltf` requires the following build tools:

- g++ 10.3.0 or above 
- cmake 3.12.4 or above

:code:`rmltf` depends on the following libraries:

- CUDA 10.1 & cuDNN 7.6.5
- OpenMPI 4.1.1
- HDF5 1.10.5
- Boost 1.76.0 (header only)
- libtensorflow 2.3.4 (pre-compiled)
- cppflow 2.0 (header only)

Library installation
--------------------

Installation instructions for all libraries *rmltf* depends on are given below.
This process was completed successfully on Ubuntu 18.04, but should be repeatable without
major changes in other Unix-like operating systems.

CUDA and cuDNN
~~~~~~~~~~~~~~

- Versions: CUDA 10.1 and cuDNN 7.6.5
- Installation steps:

.. code-block:: bash

    sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb 
    sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda

    sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb 
    sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb 
    sudo dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.1_amd64.deb 

- Download:
    https://developer.nvidia.com/cuda-downloads
    https://developer.nvidia.com/rdp/cudnn-download

OpenMPI
~~~~~~~

- Minimum version: 4.1.1
- Installation steps:

.. code-block:: bash

    tar xvzf openmpi-4.1.1.tar.gz
    cd openmpi-4.1.1
    ./configure --prefix=/usr/local --enable-mpi-cxx
    make -j8
    sudo make install
    sudo ldconfig

- Download:
    https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz

HDF5
~~~~

- Minumum version: 1.10.5
- Installation steps:

.. code-block:: bash

    tar xvzf hdf5-1.10.5.tar.gz
    cd hdf5-1.10.5/
    CC=/usr/local/bin/mpicc ./configure --enable-parallel --prefix=/usr/local/
    make -j8
    sudo make install

- Download:
    https://www.hdfgroup.org/package/hdf5-1-10-5-tar-gz/?wpdmdl=13571&refresh=6201b1de6d2411644278238

Boost
~~~~~

- Minumum version: 1.76.0
- Installation steps (copy headers only):

.. code-block:: bash

    tar xvzf boost_1_76_0.tar.gz
    sudo mv boost_1_76_0/boost /usr/local/include/

- Download:
    https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz

libtensorflow
~~~~~~~~~~~~~

- Version: 2.3.4
- Installation steps (from pre-built library):

.. code-block:: bash

    sudo mkdir -p /usr/local/libtensorflow/
    sudo tar xvzf ./libtensorflow-gpu-linux-x86_64-2.3.4.tar.gz -C /usr/local/libtensorflow/

- Download:
    https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.3.4.tar.gz
    
cppflow
~~~~~~~

- Version: 2
- Installation steps (copy headers only):

.. code-block:: bash

    tar -xvzf cppflow.tar.gz
    sudo mv cppflow/include/cppflow/ /usr/local/include/

- Download:
    https://github.com/serizba/cppflow

Building *rmltf*
----------------

.. code-block:: bash

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/:/usr/local/libtensorflow/lib/
    cd rocketml-cpp
    mkdir build
    cd build
    cmake .. -DBUILD_TF=ON
    make rmltf -j4
    sudo make install

Licenses
--------

.. list-table::
   :widths: 50 75
   :header-rows: 1

   * - Library / Tool
     - License
   * - GNU g++ 10.3.0
     - General Public License (GPL), Ver. 3
   * - CMake 3.12.4
     - 3-clause BSD License
   * - OpenMPI 4.1.1
     - 3-clause BSD License
   * - HDF5 1.10.5
     - BSD-style licenses
   * - TensorFlow 2.3.4
     - Apache 2.0 License
   * - cppflow 2.0
     - MIT License
   * - CUDA 10.1 / CuDNN 7+
     - NVIDIA proprietary Software License Agreement
   * - Boost 1.76.0
     - Boost Software License, Ver. 1.0

