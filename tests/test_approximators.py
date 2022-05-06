from unittest import TestCase
import numpy as np

from structurednets.approximators.psm_approximator import PSMApproximator
from structurednets.approximators.sss_approximator import SSSApproximator

class LayerTests(TestCase):
    def test_psm_approximator(self):
        nnz_share = 0.5
        optim_mat = np.random.uniform(-1,1, size=(51,10))
        approximator = PSMApproximator(nb_matrices=2, linear_nb_nonzero_elements_distribution=True)
        res = approximator.approximate(optim_mat, nb_params_share=nnz_share)
        approx_mat_dense = res["approx_mat_dense"]
        self.assertTrue(np.array_equal(approx_mat_dense.shape, np.array([51, 10])), "The approximated optim_mat has the wrong shape")
        self.assertTrue(np.sum([np.sum(sparse_factor != 0) for sparse_factor in res["faust_approximation"]]) <= int(nnz_share * optim_mat.size), " Too many non zero elements")

    def test_sss_approximator(self):
        optim_mat = np.random.uniform(-1,1, size=(512,100))
        approximator = SSSApproximator(nb_states=50)
        res = approximator.approximate(optim_mat, nb_params_share=0.45)
        approx_mat_dense = res["approx_mat_dense"]
        self.assertTrue(np.array_equal(approx_mat_dense.shape, np.array([512, 100])), "The approximated optim_mat has the wrong shape")