import pickle 
import numpy as np

from structurednets.asset_helpers import get_all_test_matrix_dicts
from structurednets.approximators.hmat_approximator import HMatApproximator
from structurednets.approximators.tl_approximator import TLApproximator
from structurednets.approximators.lr_approximator import LRApproximator
from structurednets.approximators.psm_approximator_wrapper import PSMApproximatorWrapper
from structurednets.approximators.sss_approximator_wrapper import SSSApproximatorWrapper
from structurednets.models.googlenet import GoogleNet
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.resnet18 import Resnet18

def get_approximation_error(orig: np.ndarray, approx: np.ndarray) -> float:
    return np.linalg.norm(orig-approx, ord="fro")

def benchmark_approximate_test_matrices(results_filepath="test_matrices_approximation_result.p"):
    nb_params_shares = np.linspace(start=0.1, stop=0.5, num=5)
    approximators = [
        HMatApproximator(),
        TLApproximator(),
        LRApproximator(),
        PSMApproximatorWrapper(),
        SSSApproximatorWrapper(),
    ]

    test_matrix_dicts = get_all_test_matrix_dicts()
    result = dict()
    
    for nb_params_share in nb_params_shares:
        result[nb_params_share] = dict()
        for approximator in approximators:
            approximator_name = approximator.get_name()
            result[nb_params_share][approximator_name] = dict()
            for test_matrix_dict in test_matrix_dicts:
                test_matrix_name = test_matrix_dict["name"]
                result[nb_params_share][approximator_name][test_matrix_name] = dict()
                for test_matrix_idx in range(3):
                    test_matrix = test_matrix_dict[test_matrix_idx]["mat"]
                    res_dict = approximator.approximate(optim_mat=test_matrix, nb_params_share=nb_params_share)

                    error = get_approximation_error(orig=test_matrix, approx=res_dict["approx_mat_dense"])
                    nb_params = res_dict["nb_parameters"]
                    approximator_type = res_dict["type"]

                    result[nb_params_share][approximator_name][test_matrix_name][test_matrix_idx] = dict()
                    result[nb_params_share][approximator_name][test_matrix_name][test_matrix_idx]["error"] = error
                    result[nb_params_share][approximator_name][test_matrix_name][test_matrix_idx]["nb_params"] = nb_params
                    result[nb_params_share][approximator_name][test_matrix_name][test_matrix_idx]["type"] = approximator_type
                    
    pickle.dump(result, open(results_filepath, "wb"))

def benchmark_approximate_weight_matrices(results_filepath="weight_matrices_approximation_result.p"):
    nb_params_shares = np.linspace(start=0.1, stop=0.5, num=5)
    approximators = [
        HMatApproximator(),
        LRApproximator(),
        PSMApproximatorWrapper(),
        SSSApproximatorWrapper(),
    ]

    required_indices = np.arange(1000)
    model_classes = [
        GoogleNet,
        InceptionV3,
        MobilenetV2,
        Resnet18,
    ]

    model_names = [model_class.__name__ for model_class in model_classes]
    weight_matrices = [model_class(output_indices=required_indices, use_gpu=False).get_optimization_matrix().detach().numpy() for model_class in model_classes]

    result = dict()
    
    for nb_params_share in nb_params_shares:
        result[nb_params_share] = dict()
        for approximator in approximators:
            approximator_name = approximator.get_name()
            result[nb_params_share][approximator_name] = dict()
            for model_name, weight_matrix in zip(model_names, weight_matrices):
                result[nb_params_share][approximator_name][model_name] = dict()
                res_dict = approximator.approximate(optim_mat=weight_matrix, nb_params_share=nb_params_share)

                error = get_approximation_error(orig=weight_matrix, approx=res_dict["approx_mat_dense"])
                nb_params = res_dict["nb_parameters"]
                approximator_type = res_dict["type"]

                result[nb_params_share][approximator_name][model_name] = dict()
                result[nb_params_share][approximator_name][model_name]["error"] = error
                result[nb_params_share][approximator_name][model_name]["nb_params"] = nb_params
                result[nb_params_share][approximator_name][model_name]["type"] = approximator_type
                    
    pickle.dump(result, open(results_filepath, "wb"))

if __name__ == "__main__":
    benchmark_approximate_test_matrices()
    #benchmark_approximate_weight_matrices()