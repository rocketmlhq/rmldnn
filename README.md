# rmldnn
RocketML Deep Neural Networks

RMLDNN is a command-line tool to build deep learning models. To use rmldnn, you do not need to know Python and deep learning frameworks like Pytorch or Tensorflow. Using rmldnn, you can build deep learning models on a laptop with a single GPU or on a data center/supercomputer with 100s of GPUs without any prior knowledge of distributed computing.

To launch a deep-learning run one has to do the following at the command line:

`rmldnn --config=<json_config_file>`

Every possible aspect of how the run is configured must be passed in the JSON file specified with the --config command-line argument. This file controls everything from log file names to hyperparameter values, all the way to details of every layer in the network. It is composed of several sections (JSON objects) which configure different aspects of the deep-learning run (e.g., optimizer parameters, data loader type, etc), some of which are specific to the type of application being executed. More on the configuration file in [concepts](#concepts) section.

# Contents

- [Concepts](#concepts)
- [Benefits](#benefits)
- [Who is this for?](#who-is-this-for)
- [Who is this not for?](#who-is-this-not-for)
- [Install](#install)
- [Concepts](#concepts)

# Benefits

- Skip writing boilerplate Python code in Tensorflow/Pytorch and build high-performance deep learning models faster
- Designed for scalability & performance so you can focus on achieving optimal deep learning model
- Boost your productivity by using the CLI for building models for different computer vision use cases like image classification, object detection, image segmentation and autoencoders.
- Run on 1-100s of GPUs or CPUs without any knowledge of distributed computing. RMLDNN will manage CPU and GPU memory, I/O, data communication between GPUs, and other complex details.

# Who is this for?

- Researchers who are solving image classification, object detection, or image segmentation problems in their respective fields using deep learning.
- Data scientists with experience in scikit-learn and venturing into deep learning.
- Data scientists who need to scale their deep learning solution that works on a single GPU to multiple GPUs.
- Data scientists who want to solve deep learning problems without writing boilerplate code in Python/Pytorch/Tensorflow.
- Newcomers to the field of machine learning who understand deep learning basics and want to build models quickly.

# Who is this not for?

- Data scientists who can build new neural network layers, neural network architectures, loss functions, and optimizers.

# Install

- Docker
  - docker pull
- Singularity
  - singularity pull

# Concepts

To launch a deep-learning run from the command line, one has to do:

`rmldnn --config=<json_config_file>`

Every possible aspect of how the run is configured must be passed in the JSON file specified with the --config command-line argument. This file controls everything from log file names to hyperparameter values, all the way to details of every layer in the network. It is composed of several sections (JSON objects) which configure different aspects of the deep-learning run (e.g., optimizer parameters, data loader type, etc), some of which are specific to the type of application being executed. 

- 
