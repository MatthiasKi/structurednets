import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod, ABC

from structurednets.layers.layer_helpers import get_random_glorot_uniform_matrix, get_nb_model_parameters
from structurednets.approximators.lr_approximator import LRApproximator
from structurednets.training_helpers import train, train_with_decreasing_lr

class StructuredLayer(nn.Module, ABC):
    def __init__(self, input_dim: int, output_dim: int, nb_params_share: float, use_bias=True, initial_weight_matrix=None, initial_bias=None):
        super(StructuredLayer, self).__init__()
        assert input_dim > 0, "The input dim should be greater than 0"
        assert output_dim > 0, "The output dim should be greater than 0"
        if nb_params_share is not None:
            assert nb_params_share >= 0 and nb_params_share <= 1, "The nb_params_share should be between 0 and 1"
        
        self.output_dim = output_dim

        if initial_weight_matrix is not None:
            assert isinstance(initial_weight_matrix, np.ndarray), "The initial weight matrix must be passed as np.ndarray"
            assert len(initial_weight_matrix.shape) == 2, "The initial weight matrix should have 2 dimensions"
            assert np.array_equal(initial_weight_matrix.shape, np.array([output_dim, input_dim])), "The initial weight matrix does not have the expected shape (" + str(output_dim) + "," + str(input_dim) + ")"

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
    
    @abstractmethod
    def forward(self, U):
        return torch.zeros((U.shape[0], self.output_dim))
    
    @abstractmethod
    def get_nb_parameters(self) -> int:
        return 0

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