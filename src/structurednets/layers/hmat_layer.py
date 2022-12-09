import torch
import torch.nn as nn
import numpy as np

from structurednets.layers.layer_helpers import get_nb_model_parameters, get_random_glorot_uniform_matrix
from structurednets.approximators.hmat_approximator import HMatApproximator
from structurednets.layers.structured_layer import StructuredLayer
from structurednets.training_helpers import train, train_with_decreasing_lr
from structurednets.hmatrix.hmatrix_component import HMatrixComponent

class HMatLayer(StructuredLayer):
    def __init__(self, input_dim: int, output_dim: int, nb_params_share: float, use_bias=True, initial_weight_matrix=None, initial_bias=None, eta=0.5, initial_hmatrix=None):
        super(HMatLayer, self).__init__(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share, use_bias=use_bias, initial_weight_matrix=initial_weight_matrix, initial_bias=initial_bias)
        assert initial_weight_matrix is None or initial_hmatrix is None, "You can either pass an initial hmatrix or an initial weight matrix to be used as starting point fo the HMatrixLayer"

        if initial_hmatrix is not None:
            self.hmatrix = initial_hmatrix.clone()
        else:
            if initial_weight_matrix is None:
                initial_weight_matrix = get_random_glorot_uniform_matrix((output_dim, input_dim))
            hmat_approximator = HMatApproximator(eta=eta)
            res_dict = hmat_approximator.approximate(optim_mat=initial_weight_matrix, nb_params_share=nb_params_share)
            self.hmatrix = res_dict["h_matrix"].clone()

        nonempty_hmatrix_components = self.hmatrix.get_all_hmatrix_components()
        nonempty_hmatrix_components = [component for component in nonempty_hmatrix_components if component.are_low_rank_components_set()]
        self.hmatrix_components = nn.ModuleList(nonempty_hmatrix_components)
        
    def add_component_to_res(self, component: HMatrixComponent):
        self.res[component.row_range.start:component.row_range.stop,:] += torch.matmul(component.left_lr, torch.matmul(component.right_lr, self.U_T[component.col_range.start:component.col_range.stop,:]))

    def forward(self, U):
        res = torch.zeros((self.output_dim, U.shape[0]))
        U_T = U.T
        # NOTE: This could still be improved by parallelizing the component multiplications.
        # However, then we need a thread-safe combination of the results (and the trivial solution for this would consume a lot memory)
        for component in self.hmatrix_components:
            res[component.row_range.start:component.row_range.stop,:] += torch.matmul(component.left_lr, torch.matmul(component.right_lr, U_T[component.col_range.start:component.col_range.stop,:]))
        res = res.T
        
        if self.use_bias:
            res += self.bias
        
        return res
    
    def get_nb_parameters(self) -> int:
        return self.hmatrix.get_nb_params()

if __name__ == "__main__":
    input_dim = 50
    output_dim = 50
    nb_params_share = 0.5

    optim_mat = np.random.uniform(-1, 1, size=(output_dim, input_dim)).astype(np.float32)
    random_input = np.random.uniform(-1, 1, size=(10, input_dim)).astype(np.float32)
    random_input_torch = torch.tensor(random_input)

    approximator = HMatApproximator()
    res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share)
    approximator_pred = (res_dict["approx_mat_dense"] @ random_input.T).T
    layer = HMatLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share, use_bias=False, initial_hmatrix=res_dict["h_matrix"])
    layer_pred = layer.forward(random_input_torch).detach().numpy()

    # ---

    input_dim = 20
    output_dim = 20
    nb_params_share = 0.5

    nb_training_samples = 1000
    train_input = np.random.uniform(-1, 1, size=(nb_training_samples, input_dim)).astype(np.float32)
    train_input_torch = torch.tensor(train_input).float()
    train_output = np.ones((nb_training_samples, output_dim), dtype=np.float32)

    layer = HMatLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share)
    pred = layer.forward(train_input_torch).detach().numpy()
    max_error_before = np.max(np.abs(train_output - pred))

    trained_layer, start_train_loss, start_train_accuracy, start_val_loss, start_val_accuracy, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train(
        model=layer, X_train=train_input, y_train=train_output,
        patience=10, batch_size=nb_training_samples, verbose=False, lr=1e-3, 
        restore_best_model=False, loss_function_class=torch.nn.MSELoss,
        min_patience_improvement=1e6, optimizer_class=torch.optim.SGD,
    )

    pred = trained_layer.forward(train_input_torch).detach().numpy()
    max_error_after = np.max(np.abs(train_output - pred))
        
    # ---

    input_dim = 51
    output_dim = 40
    initial_weight_matrix = np.random.uniform(-1, 1, size=(output_dim, input_dim))

    nb_param_share = 0.2
    max_nb_parameters = int(nb_param_share * input_dim * output_dim)
    min_nb_parameters = int((nb_param_share - 0.1) * input_dim * output_dim)
    
    layer = HMatLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share)
    nb_params = get_nb_model_parameters(layer)

    halt = 1