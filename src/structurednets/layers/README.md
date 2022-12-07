# Structurednets: Neural Network Layers with structured weight matrices

In this folder, we provide neural network layers with structured weight matrices. These layers are implemented in PyTorch, and they can be used for training structured weight matrices from scratch - or finetuning structured matrices (which have, for example, been approximated from trained weight matrices using the approximators provided in this package). 

## Layers: Overview

There are layers for the following classes of structured matrices:
- Sequentially Semiseparable Matrices: SSSLayer
- Low Rank Matrices: LRLayer
- Matrices of Low Displacement Rank (in particular: LDR matrices with tridiagonal plus corder operator matrices): LDRLayer
- Hierarchical Matrices: HMatLayer
- Products of Sparse Matrices: PSMLayer

Note that an efficient implementation of the matrix-vector / matrix-matrix product is still missing for some layers. Instead, for example the dense matrix is reconstructed and used for compting matrix-matrix multiplications (which might be computationally expensive).

## Usage

Each layer is an implementation of the abstract mother class `StructuredLayer` (which in turn inherits from the `torch.nn.Module` class). By that, each layer is required to implement the following two functions:
- `def forward(self, U) -> torch.tensor` (Note that the input to the layer has the shape `(batch_size, input_size)`, and the shape of the output is `(batch_size, output_size)`)
- `def get_nb_parameters(self) -> int` (this function can be used to check the number of free parameters used in the layer - another approach would be to use the `get_nb_model_parameters` function defined in `structurednets.layers.layer_helpers`)

All layers have the same constructor (plus optional additional arguments for some layers):

    def __init__(self, input_dim: int, output_dim: int, nb_params_share: float, use_bias=True, initial_weight_matrix=None, initial_bias=None)

This means that all layers can be initialized the same way (starting from a given weight matrix or from randomly initialized matrices). If a structured matrix of the type used in the layer is already given, it can be passed as argument to the constructor to start from there. This is done by passing additional (optional) arguments to the constructor, which differ depending on the structure class. 

Note that in our layers, the output is computed using `W U`, where `W` is the weight matrix and `U` is the input. In contrast, the `VisionModels` defined in this package compute the output using `U W`. This is, because the layers are easier to understand and easier to implement if the structured matrix is the first matrix in the product. In practice, this does not make any difference. However, please note that you might need to transpose a weight matrix extracted from a `VisionModel` first, before using it in a `StructuredLayer`.

## Training

Some of the layers contain sparse matrices (for example, the operator matrices of the low displacement rank matrices are stored using sparse matrices). `Pytorch` currently does not support training such matrices using the `Adam` optimizer. However, the `SGD` step size optimizer (without Nesterov momentum) should work in any case. In order to improve the convergence during the training with the `SGD` optimizer, we implemented an approach for reducing the learning rate each time the validation loss reaches a plateau (which means it is not improving anymore). Before reducing the learning rate, the model with the best validation accuracy is restored. The respective function for training with this approach is defined in `structurednets.training_helpers`:

    train_with_decreasing_lr(model: torch.nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, patience=10, batch_size=1000, verbose=False, loss_function_class=torch.nn.CrossEntropyLoss, min_patience_improvement=1e-10, optimizer_class=torch.optim.SGD)

Note that if no validation data is provided, automatically 20% of the provided training data is used as validation data. 