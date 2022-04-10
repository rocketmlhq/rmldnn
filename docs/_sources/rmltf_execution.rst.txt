*rmltf* execution
=================

The :code:`rmltf` application can be launched as a single-process run with the following command:

.. code-block:: bash

    rmltf --config=<json_config_file>

In addition, :code:`rmltf` can be executed on multiple processes on a single machine, 
or on multiple processes distributed across several machines.

**Single-machine parallel execution**

In order to spawn multiple processes, one can use the :code:`mpirun` tool of the
MPI suite to launch :code:`rmltf` in parallel:

.. code-block:: bash

    mpirun -np <num_procs> [mpi_options] -x TF_CPP_MIN_LOG_LEVEL=2 rmltf --config=<json_config_file>

The TF_CPP_MIN_LOG_LEVEL environment variable is used to suppress unnecessary Tensorflow
messages to the standard output. 

1. When running on a multi-GPU machine, the environment variable CUDA_VISIBLE_DEVICES 
must be passed to :code:`mpirun` in order to assign GPU devices to each process.
For example, in order to run :code:`rmltf` on 4 GPUs, one would use the following command:

.. code-block:: bash

    mpirun -np 4 -x CUDA_VISIBLE_DEVICES=0,1,2,3 -x TF_CPP_MIN_LOG_LEVEL=2 rmltf --config=<json_config_file>

2. When running on a multi-core CPU machine, the environment variable
TF_NUM_INTRAOP_THREADS can be used to control how many cores to assign to each process. 
For example, when running on a 32-core CPU node, one might want to launch 4 processes using 8 cores each:

.. code-block:: bash

    mpirun -np 4 -x TF_NUM_INTRAOP_THREADS=8 -x TF_CPP_MIN_LOG_LEVEL=2 --bind-to none -mca btl ^openib rmltf --config=<json_config_file>

If TF_NUM_INTRAOP_THREADS is not specified, :code:`rmltf` will simply split all available cores on the machine
among the processes. This is usually a desirable configuration, unless there are already other jobs running on the 
machine, in which case one would want to limit how many cores are used by :code:`rmltf`.

**Multiple-machine parallel execution**

In order to run on multiple machines, one must use some kind of cluster management tool (e.g., slurm) to spawn 
processes across multiple nodes. The examples below use the :code:`salloc` tool of the slurm suite.
The parameter :code:`-N` specifies how many machines to use, while :code:`--ntasks-per-node` controls
how many processes to spawn on each machine.

1. When running on several multi-GPU machines, the environment variable CUDA_VISIBLE_DEVICES must be used to 
control device-to-process assignment on each node. For example, to launch 8 processes split across 4 machines
with 2 GPUs each:

.. code-block:: bash

    salloc -N 4 --ntasks-per-node 2 mpirun -x CUDA_VISIBLE_DEVICES=0,1 -x TF_CPP_MIN_LOG_LEVEL=2 -mca btl ^openib -mca btl_tcp_if_include eth0 rmltf --config=<json_config_file>

2. When running on several multi-core CPU machines, the environment variable
TF_NUM_INTRAOP_THREADS constrols how many cores to assign to each process. 
For example, to run on 4 CPU nodes with 2 processes per node, each using 16 cores:

.. code-block:: bash

    salloc -N 4 --ntasks-per-node 2 mpirun -x TF_NUM_INTRAOP_THREADS=16 -x TF_CPP_MIN_LOG_LEVEL=2 --bind-to none -mca btl ^openib -mca btl_tcp_if_include eth0 rmltf --config=<json_config_file>

 
