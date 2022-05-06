import numpy as np

from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

from structurednets.layers.semiseparablelayer import standard_dims_in_dims_out_computation, get_max_statespace_dim
from structurednets.approximators.approximator import Approximator

class SSSApproximator(Approximator):
    def __init__(self, nb_states: int):
        self.nb_states = nb_states

    def approximate(self, optim_mat: np.ndarray, nb_params_share: float):
        self.state_space_dim = get_max_statespace_dim(optim_mat=optim_mat, nb_params_share=nb_params_share, nb_states=self.nb_states)

        dims_in, dims_out = standard_dims_in_dims_out_computation(input_size=optim_mat.shape[1], output_size=optim_mat.shape[0], nb_states=self.nb_states)
        T_operator = ToeplitzOperator(optim_mat, dims_in, dims_out)
        S = SystemIdentificationSVD(toeplitz=T_operator, max_states_local=self.state_space_dim)
        system_approx = MixedSystem(S)
        approx_mat_dense = system_approx.to_matrix()

        res_dict = dict()
        res_dict["type"] = "SSSApproximator"
        res_dict["system_approx"] = system_approx
        res_dict["approx_mat_dense"] = approx_mat_dense
        res_dict["state_space_dim"] = self.state_space_dim
        res_dict["nb_states"] = self.nb_states
        return res_dict

    def get_name(self):
        return "SSSApproximator_" + str(self.state_space_dim) + "dim"

if __name__ == "__main__":
    optim_mat = np.random.uniform(-1,1, size=(4096,100))
    approximator = SSSApproximator(nb_states=100)
    res = approximator.approximate(optim_mat, nb_params_share=0.25)
    approx_mat_dense = res["approx_mat_dense"]
    assert np.array_equal(approx_mat_dense.shape, np.array([512, 100])), "The approximated optim_mat has the wrong shape"