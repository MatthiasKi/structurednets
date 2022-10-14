import numpy as np

from structurednets.approximators.approximator import Approximator
from structurednets.hmatrix.block_cluster_tree import BlockClusterTree
from structurednets.hmatrix.tree_element import TreeElement
from structurednets.hmatrix.hmatrix import HMatrix

def build_hodlr_block_cluster_tree(depth: int, matrix_shape: tuple, min_block_size=2) -> BlockClusterTree:
    root = TreeElement(children=None, row_range=range(matrix_shape[0]), col_range=range(matrix_shape[1]))
    res = BlockClusterTree(root=root)

    for _ in range(1, depth):
        res.split_hodlr_style(matrix_shape=matrix_shape, min_block_size=min_block_size)

    assert not res.do_children_in_tree_overlap(), "After constructing the HODLR block cluster tree, no children should overlap in the tree"
    assert res.contains_all_indices(matrix_shape=matrix_shape), "All indices of the matrix to be approximated should be contained in the block cluster tree"
    assert res.do_children_span_all_indices(), "The children of the tree should span all indices after constructing the HODLR block cluster tree"

    return res

class HODLRApproximator(Approximator):
    def approximate(self, optim_mat: np.ndarray, nb_params_share: float):
        best_hmatrix = None
        best_hmatrix_error = None

        for depth in range(1, 8):
            block_cluster_tree = build_hodlr_block_cluster_tree(depth=depth, matrix_shape=optim_mat.shape)
            hmatrix = HMatrix(block_cluster_tree=block_cluster_tree)
            hmatrix.find_best_leaf_approximation(optim_mat=optim_mat, nb_params_share=nb_params_share)

            curr_error = hmatrix.get_curr_error(optim_mat=optim_mat)
            if best_hmatrix_error is None \
                or curr_error < best_hmatrix_error:
                best_hmatrix_error = curr_error
                best_hmatrix = hmatrix

        res_dict = dict()
        res_dict["type"] = "HODLRApproximator"
        res_dict["h_matrix"] = best_hmatrix
        res_dict["approx_mat_dense"] = best_hmatrix.to_dense_numpy()
        return res_dict

    def get_name(self):
        return "HODLRApproximator"

if __name__ == "__main__":
    tree = build_hodlr_block_cluster_tree(4, (100, 100))
    #tree.plot()

    optim_mat = np.random.uniform(-1,1, size=(51,10))
    approximator = HODLRApproximator()
    res = approximator.approximate(optim_mat=optim_mat, nb_params_share=0.5)
    print("Approximation Error: " + str(np.linalg.norm(res["approx_mat_dense"] - optim_mat, ord="fro")))