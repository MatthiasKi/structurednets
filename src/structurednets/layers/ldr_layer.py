import torch
import torch.nn as nn
import numpy as np

from structurednets.layers.layer_helpers import get_random_glorot_uniform_matrix, get_nb_model_parameters, get_random_glorot_uniform_matrix_torch
from structurednets.approximators.ldr_approximator import LDRApproximator, init_representation_matrices_torch, build_weight_matrix_torch, get_max_ld_rank_wrt_max_nb_free_parameters
from structurednets.training_helpers import train, train_with_decreasing_lr
from structurednets.layers.structured_layer import StructuredLayer

class LDRLayer(StructuredLayer):
    def __init__(self, input_dim: int, output_dim: int, nb_params_share: float, use_bias=True, initial_weight_matrix=None, initial_bias=None):
        super(LDRLayer, self).__init__(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share, use_bias=use_bias, initial_weight_matrix=initial_weight_matrix, initial_bias=initial_bias)
        assert input_dim == output_dim, "The LDR Layer is only implemented for square weight matrices (i.e. when the number of inputs equals the number of outputs)"

        self.target_mat_shape = (self.output_dim, input_dim)
        max_nb_parameters = int(nb_params_share * input_dim * self.output_dim)
        displacement_rank = get_max_ld_rank_wrt_max_nb_free_parameters(max_nb_free_parameters=max_nb_parameters, target_shape_mat=self.target_mat_shape)

        if displacement_rank < 0:
            self.representation_matrices = None
            self.fake_optim_param = nn.Parameter(get_random_glorot_uniform_matrix_torch((1, 1)))
        else:
            if initial_weight_matrix is not None:
                assert isinstance(initial_weight_matrix, np.ndarray), "The initial weight matrix must be passed as np.ndarray"
                assert len(initial_weight_matrix.shape) == 2, "The initial weight matrix should have 2 dimensions"
                assert np.array_equal(initial_weight_matrix.shape, np.array([self.output_dim, input_dim])), "The initial weight matrix does not have the expected shape (" + str(output_dim) + "," + str(input_dim) + ")"

                ldr_approximator = LDRApproximator()
                res_dict = ldr_approximator.approximate(optim_mat=initial_weight_matrix, nb_params_share=nb_params_share)
                representation_matrices = res_dict["ldr_mat"]
            else:
                representation_matrices = init_representation_matrices_torch(shape=self.target_mat_shape, displacement_rank=displacement_rank)

            self.representation_matrices = nn.ParameterList([nn.Parameter(representation_matrix) for representation_matrix in representation_matrices])
        
    def forward(self, U):
        # NOTE: This is very inefficient - there are better algorithms for computing an LDR matrix with another matrix. However, this is not implemented yet...
        # For more details, please read the paper "Learning Compressed Transforms with Low Displacement Rank" from Thomas et al.
        
        if self.representation_matrices is None:
            return torch.zeros((U.shape[0], self.output_dim))
        else:
            weight_matrix = build_weight_matrix_torch(representation_matrices=self.representation_matrices, target_mat_shape=self.target_mat_shape)
            y_pred = torch.matmul(weight_matrix, U.T)
            y_pred = y_pred.T
            
            if self.use_bias:
                y_pred += self.bias
            
            return y_pred
    
    def get_nb_parameters(self) -> int:
        if self.representation_matrices is None:
            return 0
        else:
            res = torch.numel(self.left_lr) + torch.numel(self.right_lr)
            if self.use_bias:
                res += torch.numel(self.bias)
            return int(res)

if __name__ == "__main__":
    input_dim = 20
    output_dim = 20
    nb_params_share = 0.1

    nb_training_samples = 1000
    train_input = np.random.uniform(-1, 1, size=(nb_training_samples, input_dim)).astype(np.float32)
    train_output = np.ones((nb_training_samples, output_dim), dtype=np.float32)

    layer = LDRLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share)
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
    
    layer = LDRLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share)
    nb_params = get_nb_model_parameters(layer)

    halt = 1