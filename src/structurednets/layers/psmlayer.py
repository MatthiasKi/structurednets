import torch
import torch.nn as nn
import scipy.sparse
import numpy as np

class PSMLayer(nn.Module):
    def __init__(self, res_dict: dict, bias: np.ndarray):
        super(PSMLayer, self).__init__()

        self.bias = torch.tensor(bias)

        self.sparse_matrices = [self.scipy_csr_to_torch(mat) for mat in res_dict["faust_approximation"]]
        self.sparse_matrices = self.sparse_matrices[::-1] # Needed in order to transpose
        assert len(self.sparse_matrices) == res_dict["nb_matrices"], f"Mismatch between expected and real number of matrices in the FAUST approximation"
        
        approx_mat_dense = torch.tensor(res_dict["approx_mat_dense"]).float()
        multiplied_mat = self.forward(torch.eye(self.sparse_matrices[-1].shape[-1]).float(), apply_bias=False)
        assert torch.allclose(approx_mat_dense, multiplied_mat), f"The out-multiplied mat should match the approx_mat_dense computed during approximation; Frobenius Norm: {torch.frobenius_norm(approx_mat_dense - multiplied_mat)}"

    def scipy_csr_to_torch(self, mat: scipy.sparse.csr_matrix) -> torch.tensor:
        coo_mat = scipy.sparse.coo_matrix(mat)
        # NOTE that the resulting matrix is transposed
        return torch.sparse_coo_tensor([coo_mat.col, coo_mat.row], coo_mat.data.T, (coo_mat.shape[1], coo_mat.shape[0])).float()

    def forward(self, U, apply_bias=True):
        y_pred = torch.matmul(self.sparse_matrices[-1], U.T)
        for sparse_mat in self.sparse_matrices[:-1][::-1]:
            y_pred = torch.matmul(sparse_mat, y_pred)
        y_pred = y_pred.T
        
        if apply_bias:
            y_pred += self.bias
        
        return y_pred
    
    def get_nb_parameters(self) -> int:
        return self.get_nb_parameters_in_weight_matrix() + len(self.bias)

    def get_nb_parameters_in_weight_matrix(self) -> int:
        return sum([len(mat.coalesce().values()) for mat in self.sparse_matrices])