import torch

from structurednets.hmatrix.hmatrix_component import HMatrixComponent
from structurednets.hmatrix.hmatrix_functions import get_range_as_indices, do_index_lists_overlap, are_index_lists_equal, get_range_combinations_without_permutation

class TreeElement:
    def __init__(self, children: list, row_range: range, col_range: range):
        self.children = children
        self.row_range = row_range
        self.col_range = col_range

        self.hmatrix_component = HMatrixComponent(self.row_range, self.col_range, None, None)

    def set_hmatrix_component(self, left_lr: torch.tensor, right_lr: torch.tensor):
        assert left_lr.shape[0] == len(self.row_range), "The size of the left low rank component should match the size of the row range"
        assert right_lr.shape[1] == len(self.col_range), "The size of the right low rank component should match the size of the column range"
        assert left_lr.shape[1] == right_lr.shape[0], "The shapes of the left and right low rank components should match such that they can be multiplied"
        self.hmatrix_component = HMatrixComponent(self.row_range, self.col_range, left_lr, right_lr)

    def add_singular_value_to_approximation(self):
        assert self.is_leaf(), "Can only modify the HMatrixComponent of leaf nodes"
        self.hmatrix_component.add_singular_value_to_approximation()

    def remove_singular_value_from_approximation(self):
        assert self.is_leaf(), "Can only modify the HMatrixComponent of leaf nodes"
        self.hmatrix_component.remove_singular_value_from_approximation()

    def get_all_hmatrix_components(self) -> list:
        if self.is_leaf():
            return [self.hmatrix_component]
        else:
            return sum([child.get_all_hmatrix_components() for child in self.children], [])

    def contains_idx(self, idx: tuple) -> bool:
        assert len(idx) == 2, "Matrix indices should be two dimensional"
        return idx[0] in self.row_range \
            and idx[1] in self.col_range

    def _get_diagonal_yPos_for_col_idx(self, matrix_shape: tuple, col_idx: int) -> float:
        return col_idx * ((matrix_shape[0] - 1) / (matrix_shape[1] - 1))

    def _is_idx_below_diagonal(self, matrix_shape: tuple, idx: tuple) -> bool:
        return idx[0] >= self._get_diagonal_yPos_for_col_idx(matrix_shape=matrix_shape, col_idx=idx[1])

    def _is_idx_over_diagonal(self, matrix_shape: tuple, idx: tuple) -> bool:
        return idx[0] <= self._get_diagonal_yPos_for_col_idx(matrix_shape=matrix_shape, col_idx=idx[1])

    def is_on_diagonal(self, matrix_shape: tuple) -> bool:
        # NOTE: We do not use the matrix diagonal as returned by np.diag() here - instead, we check if the line goind from the upper left corner of the matrix to the lower right corner of the matrix
        # intersects with the spanned indices of this element
        # By that, we achieve better approximation capabilities for matrices where one dimension is wider than the other (while not affecting square matrices)

        lower_left_corner = (self.row_range[-1], self.col_range[0])
        upper_right_corner = (self.row_range[0], self.col_range[-1])
        return self._is_idx_below_diagonal(matrix_shape=matrix_shape, idx=lower_left_corner) \
            and self._is_idx_over_diagonal(matrix_shape=matrix_shape, idx=upper_right_corner)

    def get_all_indices(self) -> list:
        return get_range_as_indices(row_range=self.row_range, col_range=self.col_range)

    def is_leaf(self) -> bool:
        return self.children is None

    def do_children_overlap(self) -> bool:
        if self.is_leaf():
            return False
        else:
            children_index_combinations = get_range_combinations_without_permutation(row_range=range(len(self.children)), col_range=range(len(self.children)))
            for index_combination in children_index_combinations:
                if do_index_lists_overlap(
                    indices1=self.children[index_combination[0]].get_all_indices(),
                    indices2=self.children[index_combination[1]].get_all_indices(),
                ):
                    return True
            return False

    def recursive_check_if_children_overlap(self) -> bool:
        if self.do_children_overlap():
            return True
        
        if self.is_leaf():
            return False
        else:
            return any([child.recursive_check_if_children_overlap() for child in self.children])

    def do_children_span_all_indices(self) -> bool:
        if self.is_leaf():
            return True
        else:
            children_indices = []
            for child in self.children:
                children_indices += child.get_all_indices()
                
            return are_index_lists_equal(
                indices1=self.get_all_indices(),
                indices2=children_indices
            )

    def recursive_check_if_children_span_all_indices(self) -> bool:
        if not self.do_children_span_all_indices():
            return False
        
        if self.is_leaf():
            return True
        else:
            return all([child.recursive_check_if_children_span_all_indices() for child in self.children])

    def split_hodlr_style(self, matrix_shape: tuple, min_block_size=2):
        assert min_block_size > 1, "The minimum block size must be greater than 1"

        if self.is_leaf():
            if len(self.row_range) > 2 * min_block_size \
                and len(self.col_range) > 2 * min_block_size \
                and self.is_on_diagonal(matrix_shape=matrix_shape):

                row_range_mid_point = int((self.row_range.stop + self.row_range.start) / 2)
                col_range_mid_point = int((self.col_range.stop + self.col_range.start) / 2)

                self.children = [
                    TreeElement(None, range(self.row_range.start, row_range_mid_point), range(self.col_range.start, col_range_mid_point)),
                    TreeElement(None, range(row_range_mid_point, self.row_range.stop), range(self.col_range.start, col_range_mid_point)),
                    TreeElement(None, range(self.row_range.start, row_range_mid_point), range(col_range_mid_point, self.col_range.stop)),
                    TreeElement(None, range(row_range_mid_point, self.row_range.stop), range(col_range_mid_point, self.col_range.stop)),
                ]
        else:
            for child in self.children:
                child.split_hodlr_style(matrix_shape=matrix_shape, min_block_size=min_block_size)

    def get_max_nb_of_children(self):
        if self.children is None:
            return 0
        else:
            return max([child.get_max_nb_of_children() for child in self.children] + [len(self.children)])

    def get_all_leaf_ranges(self) -> list:
        if self.is_leaf():
            return [(self.row_range, self.col_range)]
        else:
            return sum([child.get_all_leaf_ranges() for child in self.children], [])

    def get_all_leaf_elements(self) -> list:
        if self.is_leaf():
            return [self]
        else:
            return sum([child.get_all_leaf_elements() for child in self.children], [])

    def get_nb_params(self) -> int:
        if self.is_leaf():
            return self.hmatrix_component.get_nb_params()
        else:
            return sum([child.get_nb_params() for child in self.children])

    def get_all_elements_where_parameters_can_be_added(self, parameters_left: int) -> list:
        if self.is_leaf():
            if self.hmatrix_component.is_low_rank() \
                and self.hmatrix_component.get_nb_parameters_added_if_low_rank_component_increased() <= parameters_left:
                return [self]
            else:
                return []
        else:
            return sum([child.get_all_elements_where_parameters_can_be_added(parameters_left=parameters_left) for child in self.children], [])