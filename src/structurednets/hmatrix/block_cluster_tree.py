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
        self.root.split_hodlr_style(matrix_shape=matrix_shape, min_block_size=min_block_size)

    def get_max_nb_of_children(self):
        return self.root.get_max_nb_of_children()

    def is_binary(self):
        return self.get_max_nb_of_children() == 2

    def get_all_leaf_ranges(self):
        return self.root.get_all_leaf_ranges()

    def get_all_leaf_elements(self):
        return self.root.get_all_leaf_elements()

    def get_all_hmatrix_components(self):
        return self.root.get_all_hmatrix_components()

    def get_nb_params(self):
        return self.root.get_nb_params()

    def get_all_elements_where_parameters_can_be_added(self, max_nb_parameters: int):
        parameters_left = max_nb_parameters - self.get_nb_params()
        return self.root.get_all_elements_where_parameters_can_be_added(parameters_left=parameters_left)

    def plot(self):
        _, ax = plt.subplots()
        cmap = matplotlib.cm.get_cmap('Paired')

        leaf_ranges = self.get_all_leaf_ranges()

        max_row_idx = 0
        max_col_idx = 0
        for leaf_range in leaf_ranges:
            left_lower_pos = (leaf_range[1].start, -leaf_range[0].start)
            width = leaf_range[1].stop - leaf_range[1].start
            height = - (leaf_range[0].stop - leaf_range[0].start)

            ax.add_patch(
                Rectangle(
                    left_lower_pos, 
                    width,
                    height,
                    alpha=np.random.uniform(0.3, 0.8),
                    facecolor=cmap(np.random.uniform(0, 1))
                )
            )

            max_row_idx = max(leaf_range[0].stop, max_row_idx)
            max_col_idx = max(leaf_range[1].stop, max_col_idx)

        plt.xlim([0, max_col_idx])
        plt.ylim([-max_row_idx, 0])
        plt.show()

if __name__ == "__main__":
    pass