import torch
import torch.nn as nn
import numpy as np
import pickle

from structurednets.layers.layer_helpers import get_random_glorot_uniform_matrix, get_nb_model_parameters
from structurednets.approximators.tl_approximator import TLApproximator
from structurednets.training_helpers import train, train_with_decreasing_lr
from structurednets.layers.structured_layer import StructuredLayer

def comp_Zf_krylov_torch(f: torch.tensor, vec: torch.tensor) -> torch.tensor:
    # NOTE: We use this approach at the torch side (instead of computing the result explicitly using powers of Z_f), because this keeps the gradients smaller
    # TODO: Even with that, the backward computation is still too costly...
    res = torch.empty(size=(vec.shape[0], vec.shape[0]))
    for i in range(vec.shape[0]):
        res[:, i] = torch.roll(vec, i, dims=0)
        res[:i, i] *= f
    return res

class TLLayer(StructuredLayer):
    def __init__(self, input_dim: int, output_dim: int, nb_params_share: float, use_bias=True, initial_weight_matrix=None, initial_bias=None, initial_lr_matrices=None):
        super(TLLayer, self).__init__(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share, use_bias=use_bias, initial_weight_matrix=initial_weight_matrix, initial_bias=initial_bias)
        assert input_dim == output_dim, "The TL Layer is only implemented for square weight matrices (i.e. when the number of inputs equals the number of outputs)"
        assert initial_weight_matrix is None or initial_lr_matrices is None, "Either pass an initial weight matrix or initial low rank matrices - not both"
        if initial_lr_matrices is not None:
            assert len(initial_lr_matrices) == 2, "Expect the length of the list of initial low rank matrices to be 2"

        self.target_mat_shape = (self.output_dim, input_dim)

        if initial_lr_matrices is not None:
            G = pickle.loads(pickle.dumps(initial_lr_matrices[0]))
            H = pickle.loads(pickle.dumps(initial_lr_matrices[1]))
        elif initial_weight_matrix is not None:
            assert isinstance(initial_weight_matrix, np.ndarray), "The initial weight matrix must be passed as np.ndarray"
            assert len(initial_weight_matrix.shape) == 2, "The initial weight matrix should have 2 dimensions"
            assert np.array_equal(initial_weight_matrix.shape, np.array([self.output_dim, input_dim])), "The initial weight matrix does not have the expected shape (" + str(output_dim) + "," + str(input_dim) + ")"

            ldr_approximator = TLApproximator()
            res_dict = ldr_approximator.approximate(optim_mat=initial_weight_matrix, nb_params_share=nb_params_share)
            G = res_dict["G"]
            H = res_dict["H"]
        else:
            max_nb_parameters = int(nb_params_share * input_dim * output_dim)
            displacement_rank = int(max_nb_parameters / (input_dim + output_dim))
            G = get_random_glorot_uniform_matrix((output_dim, displacement_rank))
            H = get_random_glorot_uniform_matrix((displacement_rank, input_dim))

        self.G = nn.Parameter(torch.tensor(G).float())
        self.H = nn.Parameter(torch.tensor(H).float())

    def forward(self, U):
        # NOTE: This is very inefficient - there are better algorithms for computing an the product between a Toeplitz-like matrix with another matrix. However, this is not implemented yet...
        # For more details, please read the paper "Structured Transforms for Small-Footprint Deep Learning" from Sindhwani et al.
        # Even if the class of Toeplitz-like matrices is contained in the TD-LDR matrices (implemented in the LDRLayer), the graphs for optimizing this
        # layer are way smaller - and hence it can also be used for bigger matrices. 

        displacement_rank = self.G.shape[1]
        if displacement_rank > 0:
            weight_matrix = 0.5 * sum([comp_Zf_krylov_torch(f=1, vec=self.G[:, j]) @ comp_Zf_krylov_torch(f=-1, vec=torch.flipud(self.H[j, :])) for j in range(displacement_rank)])
            y_pred = torch.matmul(weight_matrix, U.T)
            y_pred = y_pred.T
        else:
            y_pred = torch.zeros((U.shape[0], self.output_dim))
        
        if self.use_bias:
            y_pred += self.bias
        
        return y_pred
    
    def get_nb_parameters(self) -> int:
        res = torch.numel(self.G) + torch.numel(self.H)
        if self.use_bias:
            res += torch.numel(self.bias)
        return int(res)

if __name__ == "__main__":
    input_dim = 1000
    output_dim = input_dim
    nb_params_share = 0.2

    nb_training_samples = 1000
    train_input = np.random.uniform(-1, 1, size=(nb_training_samples, input_dim)).astype(np.float32)
    train_output = np.ones((nb_training_samples, output_dim), dtype=np.float32)

    layer = TLLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share)
    trained_layer, start_train_loss, start_train_accuracy, start_val_loss, start_val_accuracy, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train_with_decreasing_lr(
        model=layer, X_train=train_input, y_train=train_output,
        patience=1, batch_size=nb_training_samples, verbose=False,
        loss_function_class=torch.nn.MSELoss,
    )

    train_input_torch = torch.tensor(train_input).float()
    pred = trained_layer.forward(train_input_torch).detach().numpy()

    max_error = np.max(np.abs(pred - train_output))
        
    # ---

    input_dim = 51
    output_dim = 40
    initial_weight_matrix = np.random.uniform(-1, 1, size=(output_dim, input_dim))

    nb_param_share = 0.2
    max_nb_parameters = int(nb_param_share * input_dim * output_dim)
    min_nb_parameters = int((nb_param_share - 0.1) * input_dim * output_dim)
    
    layer = TLLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share)
    nb_params = get_nb_model_parameters(layer)

    halt = 1