# Structured Neural Networks

## Motivation

In recent years, it has been shown that neural networks can be used to solve very complex problems like generating text, images or music, beating the best human player in Go, and achieving remarkable results in image recognition. However, in order for this progress to be achieved, ever larger and deeper networks were needed. This is especially problematic for applications targeting mobile platforms or embedded hardware like microcontrollers.

One approach to tackle this problem is to use structured matrices as weight matrices in neural networks. By that, the computational and storage cost can be reduced to the subquadratic domain. This results in faster inference, fewer required memory as well as reduced energy consumption.

In this repo, the focus is on two specific structure classes, namely Sequentially Semiseparable (SSS) Matrices and Products of Sparse Matrices. The algorithms used for finding these structures in trained weight matrices, as well as enforcing such structres during training are available in this repository. 

## Algorithms

This repository contains several algorithms which facilitate the use of structured matrices in neural networks. These include
- Approximation of trained weight matrices with structured matrices (used for approximating pretrained pytorch vision models such as AlexNet, Resnet18 or Googlenet)
- Fine-tuning structured matrices to be used in pretrained pytorch vision models
- Training structured matrices using the backpropagation algorithm

## Usage

The algorithms in this repository are tailored to pytorch vision models, which are used for image recognition on the Imagenet dataset. However, the algorithms should be applicable also to other models and datasets (which might required some modifications).

### Installation

Install the package with

    python3 -m pip install -e .

The "-e" flag makes the installation editable, which means that if you pull updates from the repo you don't need to run the installation script again.

Note that a python version >=3.8 is required.

### Running Tests

Run the unit tests (to check that everything works properly) with

    python3 setup.py test

(from the root of the package where the setup.py script lies).

### Models

You can import wrappers for the pretrained pytorch models, for example

    from structurednets.models import AlexNet
    model = AlexNet(indices)

where indices denotes the indices of the classes used (we selected subgroups of the imagenet classes to reduce computational costs for training).

These model wrappers provide functions to get the weight matrix to be optimized (get_optimization_matrix) or get the feature outputs of the model (get_features_for_batch), i.e. the outputs of the model without the last fully connected layer, which we aim to substitute with a layer containing a structured weight matrix. 

### Extracting Features

Since we only want to change the last (densely connected) layer of the deep pretrained pytorch models, we can freeze the first layers of the model and don't need to compute them over and over again during the training. In fact, we only need to compute the activations of the network at hand once, up to the layer which we want to modify. We call the inputs to the layer we want to modify "features". In order to compute these features, the last (densely connected) layer is removed from the network and the activations are stored into a file. The extraction can be done using the extract_features() function in structurednets.extract_features. The feature extraction can afterwards be checked using the check_features() function in structurednets.check_feature_extraction. 

Multiplying the features with the weight matrix of the last layer (which we call "optimization matrix") plus adding the biases of this last layer should yield the same output as propagating information through the whole network. This identity is checked in one of the unit tests (see "test_models.py").

### Training Algorithms

The sss_model_training.py script shows how a neural network which contains an SSS can be trained. The layer containing the SSS matrix is defined in layers/semiseparablelayer.py. It is trained using the Backpropagation through states algorithm.

### Approximation Algorithms

We provide 2 approximation methods:
- approximators/psm_approximator.py defines an approximator which approximates a given weight matrix by a product of sparse matrices (with defined number of nonzero elements). This class uses the PALM algorithm provided by the pyFaust package. 
- approximators/sss_approximator.py defines an approximator which performs balanced model reduction for a given weight matrix (with defined number of parameters). This class uses the tvsclib package. 

The script approximate_optim_matrix.py shows how to use these approximators to approximate the last layer of a pretrained pytorch vision model. 

### Speed Comparison

In order to have a fair speed comparison, we implemented the computational steps for evaluating the last layer in a C script (which can be found in the speed_comparison folder). The script can be compiled by running

	gcc -o run run.c

and executed by

	./run

from the folder where it has been compiled (note that the code is executed on a single core). 

The script requires a txt file (layer_description.txt), which defines the layer to be compared. This file can be generated using functions in the speed_comparison/prepare_run.py script. This script also shows how a speed benchmark can be executed, which compares the computation duration for evaluating the last layer for different number of parameters in the layer. 

### Support

Please write a git issue if you encounter any problems. We also appreciate pull requests!
