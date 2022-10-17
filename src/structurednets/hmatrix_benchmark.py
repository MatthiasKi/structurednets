import numpy as np
import pickle

from structurednets.models.googlenet import GoogleNet
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.approximators.hodlr_approximator import HODLRApproximator
from structurednets.approximators.sss_approximator_wrapper import SSSApproximatorWrapper
from structurednets.approximators.lr_approximator import LRApproximator

def get_error(approx_mat_dense: np.ndarray, optim_mat: np.ndarray):
    return np.linalg.norm(approx_mat_dense - optim_mat, ord="fro")

if __name__ == "__main__":
    # Hyperparameters
    output_indices = np.arange(1000) # This means all Imagenet classes
    model_classes = [GoogleNet, MobilenetV2, InceptionV3]
    param_shares = np.logspace(0.01, 0.9, num=10)
    output_path = "hmatrix_benchmark.p"
    # ---

    approximators = [
        HODLRApproximator(),
        SSSApproximatorWrapper(),
        LRApproximator()
    ]

    res = dict()
    for model_class in model_classes:
        model = model_class(output_indices=output_indices, use_gpu=False)
        model_name = model_class.__name__
        optim_mat = model.get_optimization_matrix().detach().numpy()

        res[model_name] = dict()
        res[model_name]["nb_parameters"] = optim_mat.size

        for param_share in param_shares:
            res[model_name][param_share] = dict()

            for approximator in approximators:
                res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=param_share)
                res[model_name][param_share][approximator.get_name()] = res_dict
                res[model_name][param_share][approximator.get_name()]["error"] = get_error(approx_mat_dense=res_dict["approx_mat_dense"], optim_mat=optim_mat)

            pickle.dump(res, open(output_path, "wb"))