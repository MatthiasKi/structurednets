import torch
import torch.nn as nn
import numpy as np

from structurednets.layers.layer_helpers import get_random_glorot_uniform_matrix, get_nb_model_parameters
from structurednets.approximators.lr_approximator import LRApproximator
from structurednets.layers.structured_layer import StructuredLayer
from structurednets.training_helpers import train, train_with_decreasing_lr

class LRLayer(StructuredLayer):
    def __init__(self, input_dim: int, output_dim: int, nb_params_share: float, use_bias=True, initial_weight_matrix=None, initial_bias=None, initial_lr_components=None):
        super(LRLayer, self).__init__(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share, use_bias=use_bias, initial_weight_matrix=initial_weight_matrix, initial_bias=initial_bias)

        assert initial_weight_matrix is None or initial_lr_components is None, "Either pass an initial weight matrix or initial lr components - not both"
        if initial_lr_components is not None:
            assert len(initial_lr_components) == 2, "Need to pass 2 lr components"
            assert initial_lr_components[0].shape[0] == output_dim and initial_lr_components[1].shape[1] == input_dim, "The provided lr components do not match the given input and output dimensions"
            assert initial_lr_components[0].shape[1] == initial_lr_components[1].shape[0], "The provided lr component shapes do not match - they can not be multiplied with each other"

        max_nb_parameters = int(nb_params_share * input_dim * output_dim)
        rank = int(max_nb_parameters / (input_dim + output_dim))

        if initial_lr_components is not None:
            self.left_lr = torch.tensor(initial_lr_components[0])
            self.right_lr = torch.tensor(initial_lr_components[1])
        elif initial_weight_matrix is not None:
            lr_approximator = LRApproximator()
            res_dict = lr_approximator.approximate(optim_mat=initial_weight_matrix, nb_params_share=nb_params_share)
            self.left_lr = torch.tensor(res_dict["left_mat"])
            self.right_lr = torch.tensor(res_dict["right_mat"])
        else:
            self.left_lr = torch.tensor(get_random_glorot_uniform_matrix((output_dim, rank)))
            self.right_lr = torch.tensor(get_random_glorot_uniform_matrix((rank, input_dim)))

        self.left_lr = nn.Parameter(self.left_lr.float())
        self.right_lr = nn.Parameter(self.right_lr.float())
        
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
    input_dim = 31
    output_dim = 20
    nb_params_share = 0.5

    nb_training_samples = 1000
    train_input = np.random.uniform(-1, 1, size=(nb_training_samples, input_dim)).astype(np.float32)
    train_output = np.ones((nb_training_samples, output_dim), dtype=np.float32)

    layer = LRLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share)
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
    
    layer = LRLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share)
    nb_params = get_nb_model_parameters(layer)

    halt = 1