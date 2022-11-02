import torch
import numpy as np
import pickle

from structurednets.hmatrix.block_cluster_tree import BlockClusterTree

class HMatrix:
    def __init__(self, block_cluster_tree: BlockClusterTree, shape=None):
        self.block_cluster_tree = block_cluster_tree
        if shape is not None:
            self.shape = shape

    def to_dense(self) -> torch.tensor:
        assert hasattr(self, "shape"), "The approximation must be performed before calling to_dense()"
        
        res = torch.zeros(self.shape)
        components = self.block_cluster_tree.get_all_hmatrix_components()
        for component in components:
            res[component.row_range.start:component.row_range.stop, component.col_range.start:component.col_range.stop] = component.to_dense()

        return res

    def to_dense_numpy(self) -> np.ndarray:
        return self.to_dense().detach().numpy()

    def get_curr_error(self, optim_mat: np.ndarray):
        return np.linalg.norm(optim_mat - self.to_dense_numpy(), ord="fro")

    def get_nb_params(self) -> int:
        return self.block_cluster_tree.get_nb_params()
    
    def get_all_hmatrix_components(self) -> list:
        return self.block_cluster_tree.get_all_hmatrix_components()

    def find_best_leaf_approximation(self, optim_mat: np.ndarray, nb_params_share: float):
        self.shape = optim_mat.shape
        max_nb_parameters = int(self.shape[0] * self.shape[1] * nb_params_share)

        leaf_elements = self.block_cluster_tree.get_all_leaf_elements()
        for leaf_element in leaf_elements:
            full_component = optim_mat[leaf_element.row_range.start:leaf_element.row_range.stop, leaf_element.col_range.start:leaf_element.col_range.stop]
            leaf_element.hmatrix_component.set_full_component(full_component)

        elements_where_parameters_can_be_added = self.block_cluster_tree.get_all_elements_where_parameters_can_be_added(max_nb_parameters=max_nb_parameters)
        while len(elements_where_parameters_can_be_added) > 0:
            best_error_reduction = None
            best_element_to_add_parameters = None

            for element_where_parameters_can_be_added in elements_where_parameters_can_be_added:
                curr_element_error_reduction = element_where_parameters_can_be_added.get_error_reduction_for_adding_a_singular_value(optim_mat=optim_mat, cache_result=True)
                if best_error_reduction is None \
                    or curr_element_error_reduction > best_error_reduction:
                    best_error_reduction = curr_element_error_reduction
                    best_element_to_add_parameters = element_where_parameters_can_be_added

            if best_element_to_add_parameters is None:
                break
            else:
                best_element_to_add_parameters.add_singular_value_to_approximation()

            elements_where_parameters_can_be_added = self.block_cluster_tree.get_all_elements_where_parameters_can_be_added(max_nb_parameters=max_nb_parameters)

    def clear_full_rank_parts_and_cached_values(self):
        self.block_cluster_tree.clear_full_rank_parts_and_cached_values()

    def clone(self):
        return pickle.loads(pickle.dumps(self))