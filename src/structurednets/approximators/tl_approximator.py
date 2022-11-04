import numpy as np
from scipy.sparse import csr_array
from scipy.linalg import toeplitz

from structurednets.approximators.approximator import Approximator
from structurednets.approximators.lr_approximator import LRApproximator
    
def get_f_circulant_matrix(size: int, f: float) -> np.ndarray:
    Z_f = np.diag(np.ones((size-1,)), k=-1)
    Z_f[0, -1] = f
    Z_f_sparse = csr_array(Z_f)
    return Z_f_sparse

def comp_krylov(Z_f: np.ndarray, vec: np.ndarray) -> np.ndarray:
    res = np.empty(shape=(Z_f.shape[0], Z_f.shape[1]))
    res[:, 0] = vec
    for i in range(1, Z_f.shape[1]):
        res[:, i] = Z_f @ res[:, i-1]
    return res

def build_tl_matrix(G: np.ndarray, H: np.ndarray) -> np.ndarray:
    optim_mat_size = G.shape[0]
    displacement_rank = G.shape[1]

    A = get_f_circulant_matrix(optim_mat_size, 1)
    B = get_f_circulant_matrix(optim_mat_size, -1)

    res = sum([comp_krylov(Z_f=A, vec=G[:, j]) @ comp_krylov(Z_f=B, vec=np.flipud(H[j, :])) for j in range(displacement_rank)])
    return 0.5 * res

class TLApproximator(Approximator):
    # This is an approximator for Toeplitz-Like Matrices. Note that the TD-LDR approximator is more powerful (it includes the toeplitz-like matrices).
    # However, the Toeplitz-Like approximation can be done analytically, without using gradient-descent (which involves the evaluation of very big
    # Jacobi matrices). Therefore, this approximator can also be used for large matrices, whereas the TD-LDR approximator would consume too much memory.
    
    def get_name(self):
        return "TLApproximator"

    def approximate(self, optim_mat: np.ndarray, nb_params_share: float):
        assert len(optim_mat.shape) == 2 and optim_mat.shape[0] == optim_mat.shape[1], "Can only handle square matrices for LDR approximation"
        optim_mat_size = optim_mat.shape[0]

        A = get_f_circulant_matrix(optim_mat_size, 1)
        B = get_f_circulant_matrix(optim_mat_size, -1)

        displacements = A @ optim_mat - optim_mat @ B
        lr_approximator = LRApproximator()
        lr_res_dict = lr_approximator.approximate(optim_mat=displacements, nb_params_share=nb_params_share)
        
        G = lr_res_dict["left_mat"]
        H = lr_res_dict["right_mat"]
        approx_mat_dense = build_tl_matrix(G=G, H=H)
        
        res_dict = dict()
        res_dict["type"] = "TLApproximator"
        res_dict["G"] = G
        res_dict["H"] = H
        res_dict["approx_mat_dense"] = approx_mat_dense
        res_dict["nb_parameters"] = lr_res_dict["nb_parameters"]

        return res_dict

if __name__ == "__main__":
    approximator = TLApproximator()

    # ---

    size = 60
    column = np.random.uniform(-1, 1, size=(size,))
    row = np.random.uniform(-1, 1, size=(size,))
    row[0] = column[0]
    optim_mat = toeplitz(c=column, r=row)

    res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=0.1)
    norm_difference = np.linalg.norm(optim_mat - res_dict["approx_mat_dense"], ord="fro")
    max_difference = np.max(np.abs(optim_mat - res_dict["approx_mat_dense"]))
    allclose = np.allclose(optim_mat, res_dict["approx_mat_dense"])
    print("Norm Difference: " + str(norm_difference))

    # ---

    mat_size = 30
    displacement_rank = 3
    G = np.random.uniform(-1, 1, size=(mat_size, displacement_rank))
    H = np.random.uniform(-1, 1, size=(displacement_rank, mat_size))
    optim_mat = build_tl_matrix(G=G, H=H)
    res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=0.2)
    max_difference = np.max(np.abs(optim_mat - res_dict["approx_mat_dense"]))
    allclose = np.allclose(optim_mat, res_dict["approx_mat_dense"])

    # ---

    optim_mat = np.random.uniform(-1, 1, size=(100, 100))
    res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=0.4)
    norm_difference = np.linalg.norm(optim_mat - res_dict["approx_mat_dense"], ord="fro")
    print("Norm Difference: " + str(norm_difference))

    halt = 1