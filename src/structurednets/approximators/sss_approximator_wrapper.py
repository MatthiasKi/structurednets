import numpy as np

from structurednets.approximators.approximator import Approximator
from structurednets.approximators.sss_approximator import SSSApproximator

class SSSApproximatorWrapper(Approximator):
    def __init__(self, num_states_steps=5):
        self.num_states_steps = num_states_steps

    def approximate(self, optim_mat: np.ndarray, nb_params_share: float):
        best_res_dict = None
        best_error = None

        smallest_dim = min(optim_mat.shape)
        nb_states_steps = np.linspace(start=smallest_dim/10, stop=smallest_dim/2, num=self.num_states_steps)
        nb_states_steps = nb_states_steps.astype("int")
        for nb_states in nb_states_steps:
            sss_approximator = SSSApproximator(nb_states=nb_states)
            res_dict = sss_approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share)
            curr_error = np.linalg.norm(optim_mat - res_dict["approx_mat_dense"], ord="fro")

            if best_res_dict is None \
                or curr_error < best_error:
                best_res_dict = res_dict
                best_error = curr_error

        return best_res_dict

    def get_name(self):
        return "SSSApproximatorWrapper"

if __name__ == "__main__":
    optim_mat = np.random.uniform(-1,1, size=(20,16))
    approximator = SSSApproximatorWrapper(num_states_steps=3)
    res = approximator.approximate(optim_mat, nb_params_share=0.2)
    approx_mat_dense = res["approx_mat_dense"]
    assert np.array_equal(approx_mat_dense.shape, np.array([20,16])), "The approximated optim_mat has the wrong shape"