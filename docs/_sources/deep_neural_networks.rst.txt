Deep Neural Networks
====================

All deep-learning applications in RocketML are built into an executable called `rmldnn`. To launch a deep-learning run from the 
command line, one has to do:

.. code-block:: bash

    rmldnn [--app=<application>] --config=<json_config_file>

Every possible aspect of how the run is configured must be passed in the *json* file specified with the :code:`--config` command-line argument.
This file controls everything from log file names to hyperparameter values, all the way to details of every layer 
in the network. It is composed of several sections (json objects) which configure different aspects of the deep-learning run (e.g., optimizer 
parameters, data loader type, etc), some of which are specific to the type of application being executed.

