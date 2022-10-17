# Structured Neural Networks

## Motivation

In recent years, it has been shown that neural networks can be used to solve very complex problems like generating text, images or music, beating the best human player in Go, and achieving remarkable results in image recognition. However, in order for this progress to be achieved, ever larger and deeper networks were needed. This is especially problematic for applications targeting mobile platforms or embedded hardware like microcontrollers.

One approach to tackle this problem is to use structured matrices as weight matrices in neural networks. By that, the computational and storage cost can be reduced to the subquadratic domain. This results in faster inference, fewer required memory as well as reduced energy consumption.

In this repo, the focus is on two specific structure classes, namely Sequentially Semiseparable (SSS) Matrices and Products of Sparse Matrices. The algorithms used for finding these structures in trained weight matrices, as well as enforcing such structres during training are available in this repository. 

The code in this repository has been used to run the experiments in our publication "Kissel and Diepold: Deep Convolutional Neural Networks with Sequentially Semiseparable Weight Matrices" accepted for presentation at the 30th European Symposium on Artificial Neural Networks (ESANN 2022). 

## Algorithms

This repository contains several algorithms which facilitate the use of structured matrices in neural networks. These include
- Approximation of trained weight matrices with structured matrices
- Fine-tuning structured matrices to be used in pretrained pytorch vision models
- Training structured weight matrices using the backpropagation algorithm

## Usage

In the following, we explain how to use this library. We use image recognition on the Imagenet dataset as use case in this repository. However, most algorithms are applicable also to other models and datasets.

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

### Extracting "Features"

The application example used in our scripts is image recognition based on the Imagenet dataset. Since this dataset is very large (it contains more than a million images, with 1000 classes in total), it is often not feasible to train a whole model again and again to compare the effect of using different matrix structures in the model. Howeve, we only focus on the last, densely connect layer of the deep pretrained models (which most often contains most of the parameters of the overall network, since parameters in the convolutional parts are shared). Therefore, we can freeze the first layers of the model and don't need to compute them over and over again during the training. 

In fact, we only need to compute the activations of the network once, up to the layer which we want to modify. We call the inputs to the layer we want to modify "features". In order to compute these features, the last (densely connected) layer is removed from the network and the activations are stored into a file. The extraction can be done using the extract_features() function in structurednets.features.extract_features. The feature extraction can afterwards be checked using the check_features() function in structurednets.features.check_feature_extraction. More details can be found in the [README.md](https://github.com/MatthiasKi/structurednets/tree/master/src/structurednets/features/README.md) file in the features folder.

Multiplying the features with the weight matrix of the last layer (which we call "optimization matrix") plus adding the biases of this last layer should yield the same output as propagating information through the whole network. This identity is checked in one of the unit tests (see "test_models.py").

### Training Algorithms

The sss_model_training.py script shows how a neural network which contains an SSS can be trained. The layer containing the SSS matrix is defined in layers/semiseparablelayer.py. It is trained using the Backpropagation through states algorithm.

### Approximation Algorithms

We provide several methods for approximating given matrices with structured matrices. These include sequentially semiseparable matrices, matrices with low displacment rank, low rank matrices, products of sparse matrices and hierarchical matrices. Please find more details in the [README.md](https://github.com/MatthiasKi/structurednets/tree/master/src/structurednets/approximators/README.md) file in the approximators folder.

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
