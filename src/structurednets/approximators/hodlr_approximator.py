import numpy as np

from structurednets.approximators.approximator import Approximator
from structurednets.hmatrix.block_cluster_tree import BlockClusterTree
from structurednets.hmatrix.tree_element import TreeElement
from structurednets.hmatrix.hmatrix import HMatrix
from structurednets.models.googlenet import GoogleNet

def build_hodlr_block_cluster_tree(depth: int, matrix_shape: tuple, min_block_size=2) -> BlockClusterTree:
    root = TreeElement(children=None, row_range=range(matrix_shape[0]), col_range=range(matrix_shape[1]))
    res = BlockClusterTree(root=root)
    for _ in range(1, depth):
        res.split_hodlr_style(matrix_shape=matrix_shape, min_block_size=min_block_size)
    res.check_validity(matrix_shape=matrix_shape)
    return res

class HODLRApproximator(Approximator):
    def approximate(self, optim_mat: np.ndarray, nb_params_share: float, max_depth=5):
        best_hmatrix = None
        best_hmatrix_error = None

        for depth in range(1, max_depth):
            block_cluster_tree = build_hodlr_block_cluster_tree(depth=depth, matrix_shape=optim_mat.shape)
            hmatrix = HMatrix(block_cluster_tree=block_cluster_tree)
            hmatrix.find_best_leaf_approximation(optim_mat=optim_mat, nb_params_share=nb_params_share)

            curr_error = hmatrix.get_curr_error(optim_mat=optim_mat)
            if best_hmatrix_error is None \
                or curr_error < best_hmatrix_error:
                best_hmatrix_error = curr_error
                best_hmatrix = hmatrix

        best_hmatrix.clear_full_rank_parts_and_cached_values()
        res_dict = dict()
        res_dict["type"] = "HODLRApproximator"
        res_dict["h_matrix"] = best_hmatrix
        res_dict["approx_mat_dense"] = best_hmatrix.to_dense_numpy()
        return res_dict

    def get_name(self):
        return "HODLRApproximator"

if __name__ == "__main__":
    tree = build_hodlr_block_cluster_tree(8, (100, 100))
    tree.plot()

    model = GoogleNet(output_indices=np.arange(1000), use_gpu=False)
    optim_mat = model.get_optimization_matrix().detach().numpy()

    approximator = HODLRApproximator()
    res = approximator.approximate(optim_mat=optim_mat, nb_params_share=0.4)
    print("Approximation Error: " + str(np.linalg.norm(res["approx_mat_dense"] - optim_mat, ord="fro")))

    hmatrix = res["h_matrix"]
    nb_hmatrix_components = len(hmatrix.block_cluster_tree.get_all_hmatrix_components())
    print("Nb Components: " + str(nb_hmatrix_components))