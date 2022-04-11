*rmldnn* installation
======================================

This section describes *rmldnn* build requirements and dependencies.

Dependencies
------------

*rmldnn* requires the following build tools:

- g++ 10.3.0 or above 
- cmake 3.12.4 or above

*rmldnn* depends on the following libraries:

- OpenMPI 4.1.1
- PETSc 3.12.1
- OpenCV 3.4.2 (requires FFMPEG)
- CUDA 10.1 & cuDNN 8+
- Boost 1.76.0 (header only)
- Libtorch 1.7.0 (pre-compiled)

Library installation
--------------------

Installation instructions for most libraries *rmldnn* depends on are given below.
This process was completed successfully on Ubuntu 18.04, but should be repeatable without
major changes in other Unix-like operating systems.

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

PETSc
~~~~~

- Minumum version: 3.12.1
- Installation steps:

.. code-block:: bash

    tar xvzf petsc-3.12.1.tar.gz
    cd petsc-3.12.1/
    ./configure --prefix=/usr/local/petsc/petsc-3.12.1-opt --with-fc=0 --with-debugging=0 COPTFLAGS='-O3 -march=native \
        -mtune=native' CXXOPTFLAGS='-O3 -march=native -mtune=native' --download-hdf5 --with-64-bit-indices
    make PETSC_DIR=../petsc-3.12.1 PETSC_ARCH=arch-linux2-c-opt all
    sudo make PETSC_DIR=../petsc-3.12.1 PETSC_ARCH=arch-linux2-c-opt install

FFMPEG
~~~~~~

- Needed by OpenCV
- Installation steps:

.. code-block:: bash

    tar xjvf ffmpeg-snapshot_N-97953-g64b1262.tar.bz2
    cd ffmpeg/
    ./configure --prefix=/usr/local/ffmpeg --extra-cflags="-fPIC" --enable-pic --enable-shared
    make -j8 
    sudo make install
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/ffmpeg/lib:/usr/local/petsc/petsc-3.12.1-opt/lib/

OpenCV
~~~~~~

- Minimum version: 3.4.2
- Installation steps:

.. code-block:: bash

    tar xvzf opencv_3.4.2.tar.gz
    cd opencv-3.4.2/
    mkdir build
    cd build
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/ffmpeg/lib/
    export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:/usr/local/ffmpeg/lib/pkgconfig
    export PKG_CONFIG_LIBDIR=${PKG_CONFIG_LIBDIR}:/usr/local/ffmpeg/lib
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_FFMPEG=ON .. 
    make -j8
    sudo make install

libtorch
~~~~~~~~

- Minimum version: 1.7.0
- Installation steps (from pre-built library):

.. code-block:: bash

    unzip libtorch-cxx11-abi-shared-with-deps-1.7.0.dev20200923_cuda10.1.zip
    sudo mkdir -p /usr/local/libtorch/1.7.0/
    sudo mv libtorch/* /usr/local/libtorch/1.7.0/

CUDA and cuDNN
~~~~~~~~~~~~~~

- Mininum versions: CUDA 10.1 and cuDNN 8.1
- Installation steps:

.. code-block:: bash

    sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
    sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
    sudo apt-get update
    sudo apt-get -y install cuda

    sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
    sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
    sudo dpkg -i libcudnn8-samples_8.1.1.33-1+cuda11.2_amd64.deb 

ndiff and h5py
~~~~~~~~~~~~~~

- Needed to run the *rmldnn* test suite
- Minimum version: 2.0
- installation steps:

.. code-block:: bash

    pip install h5py

    tar xvzf ndiff-2.00.tar.gz
    cd ndiff-2.00
    ./configure
    make
    sudo cp ndiff /usr/local/bin/

Building *rmldnn*
-----------------

.. code-block:: bash

    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/ffmpeg/lib:/usr/local/petsc/petsc-3.12.1-opt/lib/:/usr/local/lib/:/usr/local/libtorch/1.7.0/lib/
    cd rocketml-cpp
    mkdir build
    cd build
    cmake ..
    make rmldnn -j4
    sudo cp rmldnn /usr/local/bin/

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
   * - PETSc 3.12.1
     - 2-clause BSD License
   * - FFMPEG
     - GNU Lesser General Public License (LGPL) Ver. 2.1
   * - OpenCV 3.4.2
     - 3-clause BSD License
   * - Pytorch 1.7.0
     - 3-clause BSD License
   * - CUDA 10.1 / CuDNN 8+
     - NVIDIA proprietary Software License Agreement
   * - Boost 1.76.0
     - Boost Software License, Ver. 1.0

