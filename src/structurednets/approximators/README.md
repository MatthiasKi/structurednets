# Structurednets: Matrix Approximators

Often, you have a full / dense matrix requiring `O(n^2)` parameters to store the whole matrix. Moreover, the matrix-vector product also requires `O(n^2)` operations, which can be especially a problem on embedded or mobile devices.

One solution towards reducing storage as well as computational ressources is to approximate the given matrix with a structured matrix. There are many *non-sparse* structured matrices, and often we can find quite good approximations to any kind of matrix.

Depending on the problem at hand, one or the other structured matrix may be better suited for approximation. Therefore, it is benefitial to perform the approximation with respect to different matrix structures and check which one works best. An overview over the different matrix structures, and their computational complexity in terms of storage as well as required operations for computing the matrix-vector product is given in our survey paper: `Kissel and Diepold: Structured Matrices and their Application in Neural Networks: A Survey`

## Approximators: Overview

In this folder, you can find approximators for the following classes of structured matrices:
- Sequentially Semiseparable Matrices: SSSApproximator
- Low Rank Matrices: LRApproximator
- Matrices of Low Displacement Rank (in particular: LDR matrices with tridiagonal plus corder operator matrices): LDRApproximator
- Hierarchical Matrices (in particular: HODLR and HEDLR Matrices): HODLRApproximator (Hierarchically off diagonal low rank structure) and HEDLRApproximator (Hierarchically equally distributed low rank structure)
- Products of Sparse Matrices: PSMApproximator

## Usage

Each approximator is an implementation of the abstract mother class `Approximator`. By that, each approximator instance is required to implement the following two functions:
- `approximate(self, optim_mat: np.ndarray, nb_params_share: float) -> dict`
- `get_name(self) -> str`

The `dict` returned by the `approximate` function always contains an entry `approx_mat_dense`, which is the approximated matrix stored as standard dense matrix. Moreover, there is a field `type` containing a `string`, which is unique for the different approximator types (for example, for the low rank approximator, the `type` is "LRApproximator").

The other fields in the returned `dict` element are different for the different approximators. For example, the dict returned by the approximator for products of sparse matrices would contain a list of sparse matrices, whereas the hierarchical matrices approximator would contain an instance of the resulting approximated hierarchical matrix. 

## Wrappers

Some approximators require additional hyperparameters. For example, the sequentially semiseparable matrix approximator requires as input the number of stages used in the resulting approximated matrix. In order to hide these hyperparameters for easier usage in applications, we implemented some wrappers which automatically perform a hyper parameter search to find the correct parameters (in a range that we consider reasonable). To date, two wrappers exist:
- `SSSApproximatorWrapper`, which is a wrapper around the approximator for sequentially semiseparable matrices
- `PSMApproximatorWrapper`, which is a wrapper around the approximator of products of sparse matrices

However, please note that there still might be some hyperparameter search performed even in approximators which are not called `Wrapper`. This is, for example, the case in the `HODLRApproximator`, where the depth of the block cluster tree is treated as a hyperparameter. 