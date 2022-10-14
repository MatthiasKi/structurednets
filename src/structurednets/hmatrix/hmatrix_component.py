import torch
import numpy as np

class HMatrixComponent:
    def __init__(self, row_range: range, col_range: range, left_lr: torch.tensor, right_lr: torch.tensor):
        self.row_range = row_range
        self.col_range = col_range
        self.left_lr = left_lr
        self.right_lr = right_lr

    def are_low_rank_components_set(self):
        return self.left_lr is not None \
            and self.right_lr is not None

    def get_component_shape(self):
        return (self.get_row_range_size(), self.get_col_range_size())

    def get_row_range_size(self):
        return self.row_range.stop - self.row_range.start
    
    def get_col_range_size(self):
        return self.col_range.stop - self.col_range.start

    def get_nb_parameters_added_if_low_rank_component_increased(self):
        return self.get_row_range_size() + self.get_col_range_size()

    def to_dense(self):
        if self.are_low_rank_components_set():
            return torch.matmul(self.left_lr, self.right_lr)
        else:
            return torch.zeros(self.get_component_shape())

    def get_nb_params(self):
        if self.are_low_rank_components_set():
            return torch.numel(self.left_lr) + torch.numel(self.right_lr)
        else: 
            return 0

    def is_low_rank(self):
        if self.are_low_rank_components_set():
            return self.left_lr.shape[1] < self.left_lr.shape[0] \
                and self.right_lr.shape[0] < self.right_lr.shape[1]
        else:
            return True

    def set_full_component(self, full_component: np.ndarray):
        assert full_component.shape[0] == self.get_row_range_size(), "The passed full component does not match the size of the row range of this componenet"
        assert full_component.shape[1] == self.get_col_range_size(), "The passed full component does not match the size of the col range of this component"

        U, S, Vh = np.linalg.svd(full_component, full_matrices=False)
        S_root = np.sqrt(S)

        self.left_full_component = torch.tensor(U @ np.diag(S_root)).float()
        self.right_full_component = torch.tensor(np.diag(S_root) @ Vh).float()

        reassembled_full_component = torch.matmul(self.left_full_component, self.right_full_component)
        assert np.allclose(full_component, reassembled_full_component.detach().numpy(), rtol=1e-5, atol=1e-5), "After reassambling the full component it should be close to the passed full component"

    def is_full_component_set(self):
        return hasattr(self, "left_full_component") and hasattr(self, "right_full_component")

    def do_low_rank_component_shapes_match(self):
        if not self.are_low_rank_components_set():
            return True
        else:
            return self.left_lr.shape[1] == self.right_lr.shape[0]

    def add_singular_value_to_approximation(self):
        assert self.is_low_rank(), "Can only add singular values to leafs which are low rank"
        assert self.is_full_component_set(), "Can only add singular values when information about the full component had been given beforehand"
        assert self.do_low_rank_component_shapes_match(), "Mismatch between the low rank components"
        
        if self.left_lr is None:
            nb_singular_values = 1
        else:
            nb_singular_values = self.left_lr.shape[1] + 1
        
        self.left_lr = self.left_full_component[:, :nb_singular_values]
        self.right_lr = self.right_full_component[:nb_singular_values, :]

    def remove_singular_value_from_approximation(self):
        assert self.left_lr.shape[1] > 0 and self.right_lr.shape[0] > 0, "Can only remove singular values if there is at least one singular value in use"
        assert self.do_low_rank_component_shapes_match(), "Mismatch between the low rank components"
        
        if self.left_lr.shape[1] == 1:
            self.left_lr = None
            self.right_lr = None
        else:
            self.left_lr = self.left_lr[:, :-1]
            self.right_lr = self.right_lr[:-1, :]