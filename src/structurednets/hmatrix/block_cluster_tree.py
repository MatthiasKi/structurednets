import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from structurednets.hmatrix.tree_element import TreeElement
from structurednets.hmatrix.hmatrix_functions import get_range_as_indices

class BlockClusterTree:
    def __init__(self, root: TreeElement):
        self.root = root

    def _flatten_indices(self, indices: np.ndarray, dim: int) -> np.ndarray:
        return np.expand_dims(indices[dim].flatten(), axis=-1)

    def contains_all_indices(self, matrix_shape: tuple) -> bool:
        assert len(matrix_shape) == 2, "The matrix shape should be defined by two entries"

        indices = get_range_as_indices(row_range=range(matrix_shape[0]), col_range=range(matrix_shape[1]))

        for idx in indices:
            if not self.root.contains_idx(idx):
                return False
        
        return True

    def do_children_in_tree_overlap(self) -> bool:
        return self.root.recursive_check_if_children_overlap()

    def do_children_span_all_indices(self) -> bool:
        return self.root.recursive_check_if_children_span_all_indices()

    def split_hodlr_style(self, matrix_shape: tuple, min_block_size=2):
        self.root.split_into_four_children_if_applicable(matrix_shape=matrix_shape, min_block_size=min_block_size, check_if_on_diagonal=True)

    def split_hedlr_style(self, matrix_shape: tuple, min_block_size=2):
        self.root.split_into_four_children_if_applicable(matrix_shape=matrix_shape, min_block_size=min_block_size, check_if_on_diagonal=False)

    def split_wrt_eta(self, matrix_shape: tuple, eta: float, min_block_size=2) -> bool:
        return self.root.split_into_four_children_if_applicable(matrix_shape=matrix_shape, min_block_size=min_block_size, check_if_on_diagonal=False, eta=eta)

    def get_max_nb_of_children(self):
        return self.root.get_max_nb_of_children()

    def get_all_leaf_ranges(self):
        return self.root.get_all_leaf_ranges()

    def get_all_leaf_elements(self):
        return self.root.get_all_leaf_elements()

    def get_all_hmatrix_components(self):
        return self.root.get_all_hmatrix_components()

    def get_nb_params(self) -> int:
        return self.root.get_nb_params()

    def get_all_elements_where_parameters_can_be_added(self, max_nb_parameters: int):
        parameters_left = max_nb_parameters - self.get_nb_params()
        return self.root.get_all_elements_where_parameters_can_be_added(parameters_left=parameters_left)

    def clear_full_rank_parts_and_cached_values(self):
        self.root.recursively_clear_full_rank_parts_and_cached_values()

    def check_validity(self, matrix_shape: tuple):
        assert not self.do_children_in_tree_overlap(), "After constructing the HODLR block cluster tree, no children should overlap in the tree"
        assert self.contains_all_indices(matrix_shape=matrix_shape), "All indices of the matrix to be approximated should be contained in the block cluster tree"
        assert self.do_children_span_all_indices(), "The children of the tree should span all indices after constructing the HODLR block cluster tree"

    def plot(self):
        _, ax = plt.subplots()
        cmap = matplotlib.cm.get_cmap('Paired')

        hmatrix_components = self.get_all_hmatrix_components()

        max_row_idx = 0
        max_col_idx = 0
        for hmatrix_component in hmatrix_components:
            left_lower_pos = (hmatrix_component.col_range.start, -hmatrix_component.row_range.start)
            width = hmatrix_component.col_range.stop - hmatrix_component.col_range.start
            height = - (hmatrix_component.row_range.stop - hmatrix_component.row_range.start)

            ax.add_patch(
                Rectangle(
                    left_lower_pos, 
                    width,
                    height,
                    alpha=np.random.uniform(0.3, 0.8),
                    facecolor=cmap(np.random.uniform(0, 1))
                )
            )
            ax.text(left_lower_pos[0] + width / 2, left_lower_pos[1] + height / 2, str(hmatrix_component.get_current_nb_singular_values()))

            max_row_idx = max(hmatrix_component.row_range.stop, max_row_idx)
            max_col_idx = max(hmatrix_component.col_range.stop, max_col_idx)

        plt.xlim([0, max_col_idx])
        plt.ylim([-max_row_idx, 0])
        plt.show()

if __name__ == "__main__":
    pass