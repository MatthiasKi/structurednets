import torch
import torch.nn as nn
import scipy.sparse
import numpy as np

from structurednets.layers.layer_helpers import get_random_glorot_uniform_matrix
from structurednets.approximators.psm_approximator_wrapper import PSMApproximatorWrapper
from structurednets.layers.layer_helpers import get_nb_model_parameters
from structurednets.training_helpers import train, train_with_decreasing_lr
from structurednets.approximators.psm_approximator import PSMApproximator

class PSMLayer(nn.Module):
    # NOTE: Since this model contains sparse weight matrices, it can only be trained with SGD (particularly no Adam). 
    # Consider using the train_with_decreasing_lr instead of just train, since decreasing the learning rate might result in better results
    def __init__(self, input_dim: int, output_dim: int, use_bias=True, nb_params_share=None, initial_weight_matrix=None, initial_bias=None, sparse_matrices=None):
        super(PSMLayer, self).__init__()
        
        self.use_bias = use_bias
        if self.use_bias:
            if initial_bias is not None:
                self.bias = torch.tensor(initial_bias)
            else:
                self.bias = torch.tensor(get_random_glorot_uniform_matrix((output_dim,)))
            self.bias = nn.Parameter(self.bias)
        else:
            self.bias = None

        assert nb_params_share is not None or sparse_matrices is not None, "Need to pass the nb_params_share or an initial sparse matrices configuration"
        assert sparse_matrices is None or initial_weight_matrix is None, "Can either pass an initial weight matrix or an initial sparse matrices configuration"
        if sparse_matrices is None:
            if initial_weight_matrix is None:
                initial_weight_matrix = get_random_glorot_uniform_matrix((output_dim, input_dim))
            approximator = PSMApproximatorWrapper()
            res_dict = approximator.approximate(optim_mat=initial_weight_matrix, nb_params_share=nb_params_share)
            sparse_matrices = res_dict["faust_approximation"]

        self.sparse_matrices = nn.ParameterList([self.scipy_csr_to_torch(mat) for mat in sparse_matrices])

    def scipy_csr_to_torch(self, mat: scipy.sparse.csr_matrix) -> torch.tensor:
        coo_mat = scipy.sparse.coo_matrix(mat)
        res = torch.transpose(torch.sparse_coo_tensor([coo_mat.col, coo_mat.row], coo_mat.data.T, (coo_mat.shape[1], coo_mat.shape[0])).float(), 0, 1)
        res.requires_grad_(True)
        return nn.Parameter(res)

    def forward(self, U):
        y_pred = torch.sparse.mm(self.sparse_matrices[-1], U.T)
        for sparse_mat in self.sparse_matrices[:-1]:
            y_pred = torch.sparse.mm(sparse_mat, y_pred)
        y_pred = y_pred.T
        
        if self.use_bias:
            y_pred += self.bias
        
        return y_pred
    
    def get_nb_parameters(self) -> int:
        return self.get_nb_parameters_in_weight_matrix() + len(self.bias)

    def get_nb_parameters_in_weight_matrix(self) -> int:
        return sum([len(mat.coalesce().values()) for mat in self.sparse_matrices])

def build_PSMLayer_from_res_dict(res_dict: dict, bias: np.ndarray) -> PSMLayer:
    sparse_matrices = res_dict["faust_approximation"]
    assert len(sparse_matrices) == res_dict["nb_matrices"], f"Mismatch between expected and real number of matrices in the FAUST approximation"
    
    shape = res_dict["approx_mat_dense"].shape
    layer = PSMLayer(input_dim=shape[1], output_dim=shape[0], sparse_matrices=sparse_matrices, initial_bias=bias)
    layer_without_bias = PSMLayer(input_dim=shape[1], output_dim=shape[0], sparse_matrices=sparse_matrices, use_bias=False)

    approx_mat_dense = torch.tensor(res_dict["approx_mat_dense"]).float()
    multiplied_mat = layer_without_bias.forward(torch.eye(sparse_matrices[-1].shape[-1]).float())
    assert torch.allclose(approx_mat_dense, multiplied_mat), f"The out-multiplied mat should match the approx_mat_dense computed during approximation; Frobenius Norm: {torch.frobenius_norm(approx_mat_dense - multiplied_mat)}"

    return layer

if __name__ == "__main__":
    input_dim = 31
    output_dim = 20
    nb_params_share = 0.5

    nb_training_samples = 1000
    train_input = np.random.uniform(-1, 1, size=(nb_training_samples, input_dim)).astype(np.float32)
    train_output = np.ones((nb_training_samples, output_dim), dtype=np.float32)

    layer = PSMLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share)
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
    
    layer = PSMLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share)
    nb_params = get_nb_model_parameters(layer)

    halt = 1