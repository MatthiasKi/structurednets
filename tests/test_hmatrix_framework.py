from re import S
from unittest import TestCase
import numpy as np

from structurednets.hmatrix.block_cluster_tree import BlockClusterTree
from structurednets.hmatrix.tree_element import TreeElement
from structurednets.approximators.hodlr_approximator import build_hodlr_block_cluster_tree

class HMatrixFrameWorkTests(TestCase):
    def test_block_cluster_tree_with_single_element(self):
        root = TreeElement(children=None, row_range=range(10), col_range=range(10))
        tree = BlockClusterTree(root=root)

        shape1 = np.array([10, 10])
        contains_all_indices = tree.contains_all_indices(shape1)
        self.assertTrue(contains_all_indices, "All indices should be contained when passing the right shape")

        shape2 = np.array([20, 20])
        contains_all_indices = tree.contains_all_indices(shape2)
        self.assertFalse(contains_all_indices, "Not all indices should be contained when passing the wrong shape")

    def test_is_on_diagonal_function(self):
        square_shape = (10, 10)
        towel_shape = (20, 10)

        square_middle = TreeElement(children=None, row_range=range(4,6), col_range=range(4,6))
        square_left_lower_corner = TreeElement(children=None, row_range=range(8,10), col_range=(0,2))
        left_upper_corner = TreeElement(children=None, row_range=range(0,2), col_range=(0,2))
        square_right_lower_corner = TreeElement(children=None, row_range=range(8,10), col_range=(8,10))
        towel_right_lower_corner = TreeElement(children=None, row_range=(18,20), col_range=(8,10))
        towel_middle = TreeElement(children=None, row_range=range(7,9), col_range=range(3,5))
        
        self.assertTrue(square_middle.is_on_diagonal(matrix_shape=square_shape))
        self.assertFalse(square_middle.is_on_diagonal(matrix_shape=towel_shape))

        self.assertFalse(square_left_lower_corner.is_on_diagonal(matrix_shape=square_shape))
        self.assertFalse(square_left_lower_corner.is_on_diagonal(matrix_shape=towel_shape))

        self.assertTrue(left_upper_corner.is_on_diagonal(matrix_shape=square_shape))
        self.assertTrue(left_upper_corner.is_on_diagonal(matrix_shape=towel_shape))

        self.assertTrue(square_right_lower_corner.is_on_diagonal(matrix_shape=square_shape))
        self.assertFalse(square_right_lower_corner.is_on_diagonal(matrix_shape=towel_shape))

        self.assertFalse(towel_right_lower_corner.is_on_diagonal(matrix_shape=square_shape))
        self.assertTrue(towel_right_lower_corner.is_on_diagonal(matrix_shape=towel_shape))

        self.assertFalse(towel_middle.is_on_diagonal(matrix_shape=square_shape))
        self.assertTrue(towel_middle.is_on_diagonal(matrix_shape=towel_shape))

    def test_build_hodlr_block_cluster_tree_function(self):
        shapes = [(10, 10), (20, 10), (10, 20)]
        depths = range(5)
        for shape in shapes:
            for depth in depths:
                build_hodlr_block_cluster_tree(depth=depth, matrix_shape=shape)