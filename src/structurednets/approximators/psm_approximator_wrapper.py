import numpy as np

from structurednets.approximators.approximator import Approximator
from structurednets.approximators.psm_approximator import PSMApproximator

class PSMApproximatorWrapper(Approximator):
    def __init__(self, num_interpolation_steps=17, only_linear_distribution=False):
        self.num_interpolation_steps = num_interpolation_steps
        if only_linear_distribution:
            self.linear_nb_nonzero_elements_distribution_values = [True]
        else:
            self.linear_nb_nonzero_elements_distribution_values = [True, False]

    def get_name(self):
        name = "PSMApproximatorWrapper"
        return name

    def approximate(self, optim_mat: np.ndarray, nb_params_share: float):
        optim_mat64 = optim_mat.astype("float64")
        best_approximation = None
        for nb_matrices in [2, 3]:
            for linear_nb_nonzero_elements_distribution in self.linear_nb_nonzero_elements_distribution_values:
                approximator = PSMApproximator(nb_matrices=nb_matrices, linear_nb_nonzero_elements_distribution=linear_nb_nonzero_elements_distribution)
                res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share, num_interpolation_steps=self.num_interpolation_steps)
                
                if res_dict is not None:
                    norm = np.linalg.norm(optim_mat64 - res_dict["approx_mat_dense"], ord="fro")

                    if best_approximation is None or best_approximation["objective_function_result"] > norm:
                        res_dict["objective_function_result"] = norm
                        best_approximation = res_dict
        
        return best_approximation 

if __name__ == "__main__":
    nnz_share = 0.5
    optim_mat = np.random.uniform(-1,1, size=(51,10))
    approximator = PSMApproximatorWrapper()
    res = approximator.approximate(optim_mat, nb_params_share=0.5)
    approx_mat_dense = res["approx_mat_dense"]
    assert np.array_equal(approx_mat_dense.shape, np.array([51, 10])), "The approximated optim_mat has the wrong shape"
    assert np.sum([np.sum(sparse_factor != 0) for sparse_factor in res["faust_approximation"]]) <= int(nnz_share * optim_mat.size), " Too many non zero elements"