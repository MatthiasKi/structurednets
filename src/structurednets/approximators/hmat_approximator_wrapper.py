import numpy as np

from structurednets.approximators.hmat_approximator import HMatApproximator
from structurednets.approximators.approximator import Approximator

class HMatApproximatorWrapper(Approximator):
    def __init__(self, num_etas=3):
        self.num_etas=num_etas

    def approximate(self, optim_mat: np.ndarray, nb_params_share: float):
        best_res_dict = None
        best_error = None

        etas = np.linspace(start=0.5, stop=0.9, num=self.num_etas)
        for eta in etas:
            approximator = HMatApproximator(eta=eta)
            res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share)
            curr_error = np.linalg.norm(optim_mat - res_dict["approx_mat_dense"], ord="fro")

            if best_res_dict is None \
                or curr_error < best_error:
                best_res_dict = res_dict
                best_error = curr_error

        return best_res_dict

    def get_name(self):
        return "HMatApproximatorWrapper"