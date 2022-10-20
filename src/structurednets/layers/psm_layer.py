import torch
import torch.nn as nn
import scipy.sparse
import numpy as np

from structurednets.layers.layer_helpers import get_random_glorot_uniform_matrix
from structurednets.approximators.psm_approximator_wrapper import PSMApproximatorWrapper

class PSMLayer(nn.Module):
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

        self.sparse_matrices = [self.scipy_csr_to_torch(mat) for mat in sparse_matrices]
        self.sparse_matrices = self.sparse_matrices[::-1] # Needed in order to transpose        
        self.sparse_matrices = nn.ParameterList(self.sparse_matrices)
        
    def scipy_csr_to_torch(self, mat: scipy.sparse.csr_matrix) -> torch.tensor:
        coo_mat = scipy.sparse.coo_matrix(mat)
        # NOTE that the resulting matrix is transposed
        return torch.sparse_coo_tensor([coo_mat.col, coo_mat.row], coo_mat.data.T, (coo_mat.shape[1], coo_mat.shape[0])).float()

    def forward(self, U):
        y_pred = torch.matmul(self.sparse_matrices[-1], U.T)
        for sparse_mat in self.sparse_matrices[:-1][::-1]:
            y_pred = torch.matmul(sparse_mat, y_pred)
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

    approx_mat_dense = torch.tensor(res_dict["approx_mat_dense"]).float()
    multiplied_mat = layer.forward(torch.eye(sparse_matrices[-1].shape[-1]).float(), apply_bias=False)
    assert torch.allclose(approx_mat_dense, multiplied_mat), f"The out-multiplied mat should match the approx_mat_dense computed during approximation; Frobenius Norm: {torch.frobenius_norm(approx_mat_dense - multiplied_mat)}"

    return layer