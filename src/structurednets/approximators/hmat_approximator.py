import numpy as np

from structurednets.approximators.approximator import Approximator
from structurednets.hmatrix.block_cluster_tree import BlockClusterTree
from structurednets.hmatrix.tree_element import TreeElement
from structurednets.hmatrix.hmatrix import HMatrix
from structurednets.models.googlenet import GoogleNet

def build_hmat_block_cluster_tree(matrix_shape: tuple, eta: float, min_block_size=2) -> BlockClusterTree:
    assert eta > 0, "Eta must be greater than 0"
    assert eta < 1, "Eta must be smaller than 1"

    root = TreeElement(children=None, row_range=range(matrix_shape[0]), col_range=range(matrix_shape[1]))
    res = BlockClusterTree(root=root)

    while res.split_wrt_eta(matrix_shape=matrix_shape, min_block_size=min_block_size, eta=eta):
        pass

    res.check_validity(matrix_shape=matrix_shape)
    return res

class HMatApproximator(Approximator):
    def approximate(self, optim_mat: np.ndarray, nb_params_share: float, eta=0.5):
        block_cluster_tree = build_hmat_block_cluster_tree(matrix_shape=optim_mat.shape, eta=eta)
        hmatrix = HMatrix(block_cluster_tree=block_cluster_tree)
        hmatrix.find_best_leaf_approximation(optim_mat=optim_mat, nb_params_share=nb_params_share)

        hmatrix.clear_full_rank_parts_and_cached_values()
        res_dict = dict()
        res_dict["type"] = "HMatApproximator"
        res_dict["h_matrix"] = hmatrix
        res_dict["approx_mat_dense"] = hmatrix.to_dense_numpy()
        res_dict["eta"] = eta
        res_dict["nb_parameters"] = hmatrix.block_cluster_tree.get_nb_params()
        return res_dict

    def get_name(self):
        return "HMatApproximator"

if __name__ == "__main__":
    tree = build_hmat_block_cluster_tree((100, 100), 0.5)
    tree.plot()

    model = GoogleNet(output_indices=np.arange(1000), use_gpu=False)
    optim_mat = model.get_optimization_matrix().detach().numpy()

    approximator = HMatApproximator()
    res = approximator.approximate(optim_mat=optim_mat, nb_params_share=0.4)
    print("Approximation Error: " + str(np.linalg.norm(res["approx_mat_dense"] - optim_mat, ord="fro")))
    print("eta: " + str(res["eta"]))