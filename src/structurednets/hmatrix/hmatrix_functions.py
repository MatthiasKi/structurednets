import numpy as np

def get_range_as_indices(row_range: range, col_range: range) -> list:
    return [(row_idx, col_idx) for row_idx in row_range for col_idx in col_range]

def get_range_combinations_without_permutation(row_range: range, col_range: range) -> list:
    return [(row_idx, col_idx) for row_idx in row_range for col_idx in col_range if row_idx < col_idx]

def do_index_lists_overlap(indices1: list, indices2: list) -> bool:
    return not set(indices1).isdisjoint(indices2)

def are_index_lists_equal(indices1: list, indices2: list) -> bool:
    return set(indices1) == set(indices2)