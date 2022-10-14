import torch
import numpy as np

from structurednets.hmatrix.block_cluster_tree import BlockClusterTree

class HMatrix:
    def __init__(self, block_cluster_tree: BlockClusterTree):
        self.block_cluster_tree = block_cluster_tree

    def to_dense(self) -> torch.tensor:
        assert hasattr(self, "shape"), "The approximation must be performed before calling to_dense()"
        
        res = torch.zeros(self.shape)
        components = self.block_cluster_tree.get_all_hmatrix_components()
        for component in components:
            res[component.row_range, :][:, component.col_range] = component.to_dense()

        return res

    def to_dense_numpy(self) -> np.ndarray:
        return self.to_dense().detach().numpy()

    def get_curr_error(self, optim_mat: np.ndarray):
        return np.linalg.norm(optim_mat - self.to_dense_numpy(), ord="fro")

    def find_best_leaf_approximation(self, optim_mat: np.ndarray, nb_params_share: float):
        self.shape = optim_mat.shape
        max_nb_parameters = int(self.shape[0] * self.shape[1] * nb_params_share)

        leaf_elements = self.block_cluster_tree.get_all_leaf_elements()
        for leaf_element in leaf_elements:
            full_component = optim_mat[leaf_element.row_range, :][:, leaf_element.col_range]
            leaf_element.hmatrix_component.set_full_component(full_component)

        elements_where_parameters_can_be_added = self.block_cluster_tree.get_all_elements_where_parameters_can_be_added(max_nb_parameters=max_nb_parameters)
        while len(elements_where_parameters_can_be_added) > 0:
            best_error = self.get_curr_error(optim_mat=optim_mat)
            best_element_to_add_parameters = None
            for element_where_parameters_can_be_added in elements_where_parameters_can_be_added:
                element_where_parameters_can_be_added.add_singular_value_to_approximation()

                error_after_modification = self.get_curr_error(optim_mat=optim_mat)
                if error_after_modification < best_error:
                    best_error = error_after_modification
                    best_element_to_add_parameters = element_where_parameters_can_be_added

                element_where_parameters_can_be_added.remove_singular_value_from_approximation()

            if best_element_to_add_parameters is None:
                break
            else:
                best_element_to_add_parameters.add_singular_value_to_approximation()

            elements_where_parameters_can_be_added = self.block_cluster_tree.get_all_elements_where_parameters_can_be_added()

    def dot(self, vec: torch.tensor) -> torch.tensor:
        assert hasattr(self, "shape"), "The approximation must be performed before calling dot()"
        assert len(vec.shape) == 1 or vec.shape[0] == 1 or vec.shape[1] == 1, "dot is only implemented for vectors"

        # TODO implement this
        pass