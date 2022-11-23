import numpy as np

from structurednets.approximators.hmat_approximator import HMatApproximator
from structurednets.approximators.approximator import Approximator

class HMatApproximatorWrapper(Approximator):
    def __init__(self, nb_min_block_sizes=6):
        self.nb_min_block_sizes=nb_min_block_sizes

    def approximate(self, optim_mat: np.ndarray, nb_params_share: float):
        best_res_dict = None
        best_error = None

        min_block_sizes = [np.power(2, i+1) for i in range(self.nb_min_block_sizes)]
        for min_block_size in min_block_sizes:
            approximator = HMatApproximator(min_block_size=min_block_size)
            res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share)
            curr_error = np.linalg.norm(optim_mat - res_dict["approx_mat_dense"], ord="fro")

            if best_res_dict is None \
                or curr_error < best_error:
                best_res_dict = res_dict
                best_error = curr_error

        return best_res_dict

    def get_name(self):
        return "HMatApproximatorWrapper"