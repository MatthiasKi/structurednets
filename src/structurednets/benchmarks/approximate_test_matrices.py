import pickle 
import numpy as np
import time

from structurednets.asset_helpers import get_all_test_matrix_dicts
from structurednets.approximators.hmat_approximator_wrapper import HMatApproximatorWrapper
from structurednets.approximators.tl_approximator import TLApproximator
from structurednets.approximators.lr_approximator import LRApproximator
from structurednets.approximators.psm_approximator_wrapper import PSMApproximatorWrapper
from structurednets.approximators.sss_approximator_wrapper import SSSApproximatorWrapper
from structurednets.models.googlenet import GoogleNet
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.resnet18 import Resnet18
from structurednets.features.extract_features import get_required_indices
from structurednets.asset_helpers import get_all_classes_filepath

def get_approximation_error(orig: np.ndarray, approx: np.ndarray) -> float:
    return np.linalg.norm(orig-approx, ord="fro")

def benchmark_approximate_test_matrices(results_filepath="test_matrices_approximation_result.p"):
    nb_params_shares = np.linspace(start=0.1, stop=0.5, num=5)
    approximators = [
        HMatApproximatorWrapper(),
        TLApproximator(),
        LRApproximator(),
        PSMApproximatorWrapper(num_interpolation_steps=3, max_nb_matrices=2, max_last_mat_param_share=0.5),
        SSSApproximatorWrapper(num_states_steps=3),
    ]

    test_matrix_dicts = get_all_test_matrix_dicts()
    result = dict()
    
    for nb_params_share in nb_params_shares:
        print("Starting Param share " + str(nb_params_share))
        result[nb_params_share] = dict()
        for approximator in approximators:
            approximator_name = approximator.get_name()
            print("Starting approximator " + str(approximator_name))
            result[nb_params_share][approximator_name] = dict()
            for test_matrix_dict in test_matrix_dicts:
                start_time = time.time()

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
                
                elapsed_time = time.time() - start_time
                print(str(test_matrix_name) + " took " + str(elapsed_time) + " seconds (" + str(approximator_name) + ", " + str(nb_params_share) + ")")
                    
        pickle.dump(result, open(results_filepath, "wb"))

def benchmark_approximate_weight_matrices(path_to_labelfile: str,results_filepath="weight_matrices_approximation_result.p"):
    nb_params_shares = np.linspace(start=0.1, stop=0.5, num=5)
    approximators = [
        HMatApproximatorWrapper(),
        LRApproximator(),
        PSMApproximatorWrapper(num_interpolation_steps=3, max_nb_matrices=2, max_last_mat_param_share=0.5),
        SSSApproximatorWrapper(num_states_steps=3),
    ]

    required_indices = get_required_indices(path_to_label_file=path_to_labelfile)
    model_classes = [
        GoogleNet,
        InceptionV3,
        MobilenetV2,
        Resnet18,
    ]

    model_names = [model_class.__name__ for model_class in model_classes]
    # NOTE: Since in our layers, the matrix-vector multiplication is transposed compared to the VisionModels, we need to transpose the weight matrix here
    weight_matrices = [model_class(output_indices=required_indices, use_gpu=False).get_optimization_matrix().detach().numpy().T for model_class in model_classes]

    result = dict()
    
    for nb_params_share in nb_params_shares:
        print("Starting Param share " + str(nb_params_share))
        result[nb_params_share] = dict()
        for approximator in approximators:
            approximator_name = approximator.get_name()
            print("Starting approximator " + str(approximator_name))
            result[nb_params_share][approximator_name] = dict()
            for model_name, weight_matrix in zip(model_names, weight_matrices):
                start_time = time.time()

                result[nb_params_share][approximator_name][model_name] = dict()
                res_dict = approximator.approximate(optim_mat=weight_matrix, nb_params_share=nb_params_share)

                error = get_approximation_error(orig=weight_matrix, approx=res_dict["approx_mat_dense"])
                nb_params = res_dict["nb_parameters"]
                approximator_type = res_dict["type"]

                result[nb_params_share][approximator_name][model_name] = dict()
                result[nb_params_share][approximator_name][model_name]["error"] = error
                result[nb_params_share][approximator_name][model_name]["nb_params"] = nb_params
                result[nb_params_share][approximator_name][model_name]["type"] = approximator_type

                if nb_params_share == 0.2:
                    # We store this dict, since we will need it for fine tuning
                    result[nb_params_share][approximator_name][model_name]["res_dict"] = res_dict

                elapsed_time = time.time() - start_time
                print(str(model_name) + " took " + str(elapsed_time) + " seconds (" + str(approximator_name) + ", " + str(nb_params_share) + ")")
                    
    pickle.dump(result, open(results_filepath, "wb"))

if __name__ == "__main__":
    #benchmark_approximate_test_matrices()
    benchmark_approximate_weight_matrices(path_to_labelfile=get_all_classes_filepath())