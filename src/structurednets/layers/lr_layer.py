import torch
import torch.nn as nn
import numpy as np

from structurednets.layers.layer_helpers import get_random_glorot_uniform_matrix, get_nb_model_parameters
from structurednets.approximators.lr_approximator import LRApproximator

class LRLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, nb_params_share: float, use_bias=True, initial_weight_matrix=None, initial_bias=None):
        super(LRLayer, self).__init__()
        assert input_dim > 0, "The input dim should be greater than 0"
        assert output_dim > 0, "The output dim should be greater than 0"
        assert nb_params_share >= 0 and nb_params_share <= 1, "The nb_params_share should be between 0 and 1"

        max_nb_parameters = int(nb_params_share * input_dim * output_dim)
        rank = int(max_nb_parameters / (input_dim + output_dim))

        if initial_weight_matrix is not None:
            assert isinstance(initial_weight_matrix, np.ndarray), "The initial weight matrix must be passed as np.ndarray"
            assert len(initial_weight_matrix.shape) == 2, "The initial weight matrix should have 2 dimensions"
            assert np.array_equal(initial_weight_matrix.shape, np.array([output_dim, input_dim])), "The initial weight matrix does not have the expected shape (" + str(output_dim) + "," + str(input_dim) + ")"

            lr_approximator = LRApproximator()
            res_dict = lr_approximator.approximate(optim_mat=initial_weight_matrix, nb_params_share=nb_params_share)
            self.left_lr = torch.tensor(res_dict["left_mat"])
            self.right_lr = torch.tensor(res_dict["right_mat"])
        else:
            self.left_lr = torch.tensor(get_random_glorot_uniform_matrix((output_dim, rank)))
            self.right_lr = torch.tensor(get_random_glorot_uniform_matrix((rank, input_dim)))

        self.left_lr = nn.Parameter(self.left_lr.float())
        self.right_lr = nn.Parameter(self.right_lr.float())

        self.use_bias = use_bias
        if use_bias:
            if initial_bias is not None:
                assert isinstance(initial_bias, np.ndarray), "The initial bias must be passed as np.ndarray"
                assert len(initial_bias.shape) == 1, "The initial bias is expected to be passed as vector"
                assert initial_bias.shape[0] == output_dim, "The initial bias has not the expected shape (it should contain " + str(output_dim) + " values in its first dimension)"
                self.bias = torch.tensor(initial_bias)
            else:
                self.bias = torch.tensor(get_random_glorot_uniform_matrix((output_dim,)))

            self.bias = nn.Parameter(self.bias.float())
        else:
            self.bias = None
        
    def forward(self, U):
        y_pred = torch.matmul(self.right_lr, U.T)
        y_pred = torch.matmul(self.left_lr, y_pred)
        y_pred = y_pred.T
        
        if self.use_bias:
            y_pred += self.bias
        
        return y_pred
    
    def get_nb_parameters(self) -> int:
        res = torch.numel(self.left_lr) + torch.numel(self.right_lr)
        if self.use_bias:
            res += torch.numel(self.bias)
        return int(res)

if __name__ == "__main__":
    input_dim = 51
    output_dim = 40
    initial_weight_matrix = np.random.uniform(-1, 1, size=(output_dim, input_dim))


    nb_param_share = 0.2
    max_nb_parameters = int(nb_param_share * input_dim * output_dim)
    min_nb_parameters = int((nb_param_share - 0.1) * input_dim * output_dim)
    
    layer = LRLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share)
    nb_params = get_nb_model_parameters(layer)

    halt = 1