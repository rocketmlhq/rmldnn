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
- [Usage](#usage)
- [Applications](#applications)
- [Guides](#guides)
- [Troubleshooting](#troubleshooting)
- [FAQs](#faqs)
- [Citation](#citation)
- [Publications](#publications)
- [Talks](#talks)

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

![dnn_training](dnn_training.png)

As shown in the figure, a typical training process will need a data set, a deep learning model represented by a network architecture like ResNet50, U-Net etc. that is used to calculate the loss value using a function like NLL, BCE, Dice etc., back-propagation to compute gradients, and an optimizer like SGD, Adam etc. to update model weights. The json file must contain one single object named neural_network, inside which all these configurations will reside:


    {
        "neural_network": {
            "outfile": "log_file.txt",
            "num_epochs": 100,
            "data": {
                ...
            }
            "layers": {
                ...
            },
            "loss": {
                ...
            },
            "optimizer": {
                ...
            }                       
        }
    }

**Data** section is where the types of training and test data are configured, in particular, what specific data loader will be used to feed data into the neural network, as well as how that data will be split into mini-batches, how many samples will be used for training and evaluation, etc.

**Layers** section allows for detailed specification of all layers in the neural network, as well as the connections between them. The syntax is supposed to follow closely the one used by Keras, which allows exporting a programmatically built neural network as a json file.

**Loss** section specifies which loss function to use for the neural network. The loss function computes some kind of metric that estimates the error (loss) between the network result for a given input and its corresponding target. The choice of loss function must be consistent with the network design, in particular, with the last layer in the network and its activation.

**Optimizer** section configures the optimizer for the neural network, which can be selected with the parameter type. We support the most important first-order algorithms available in PyTorch (module torch.optim), as well as a Hessian-based second-order optimizer. Each optimizer type has its own set of supported hyper-parameters.


# Usage

- Single GPU
  - Docker
    ```
      docker run –gpus all -it /bin/bash
      rmldnn –config= ./<configuration_file>
    ```
  - Singularity
  singularity exec .. 

- Single node with multiple GPUs
  - Docker

- Multiple-nodes with multiple GPUs
  - Singularity
  - RocketML managed platform 
  - AWS
  - Azure

# Applications

- Image Classification
- 2D Image Segmentation
- 3D Image Segmentation
- Object Detection
- Transfer Learning
- Self-supervision
- Generative Adversarial Networks

# Guides

- RocketML
- AWS
- Azure

# Troubleshooting

Submit a ticket with as many details as possible for of your issue

# FAQs

# Citation

# Publications

# Talks

