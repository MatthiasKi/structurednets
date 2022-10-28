import pickle
import os 
import torch 

from structurednets.training_helpers import get_accuracy, get_full_batch, get_train_data, transform_feature_dtypes
from structurednets.logging_helpers import write_header_to_file, log_to_file
from structurednets.features.extract_features import get_required_indices
from structurednets.asset_helpers import get_animal_classes_filepath, load_features, get_label_name_from_path
from structurednets.models.visionmodel import VisionModel
from structurednets.models.alexnet import AlexNet
from structurednets.models.googlenet import GoogleNet
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.resnet18 import Resnet18
from structurednets.models.vgg16 import VGG16
from structurednets.approximators.psm_approximator import PSMApproximator
from structurednets.approximators.sss_approximator import SSSApproximator

def benchmark_approximate_weight_matrix(model_class: VisionModel, features_filepath: str, output_folderpath: str, label_filepath: str):
    label_tag = get_label_name_from_path(label_filepath)
    required_indices = get_required_indices(path_to_label_file=label_filepath)
    model = model_class(output_indices=required_indices, use_gpu=False)
    optim_mat_t = model.get_optimization_matrix()
    optim_mat = optim_mat_t.cpu().detach().numpy()
    bias_t = model.get_bias()
    bias = bias_t.cpu().detach().numpy()

    X_train, X_val, y_train, y_val = get_train_data(features_path=features_filepath)
    X_train, X_val, y_train, y_val = transform_feature_dtypes(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    X_train_t, y_train_t = get_full_batch(X=X_train, y=y_train)
    X_val_t, y_val_t = get_full_batch(X=X_val, y=y_val)

    approximators = [
        PSMApproximator(nb_matrices=2, linear_nb_nonzero_elements_distribution=True),
        PSMApproximator(nb_matrices=3, linear_nb_nonzero_elements_distribution=True),
        PSMApproximator(nb_matrices=2, linear_nb_nonzero_elements_distribution=False),
        PSMApproximator(nb_matrices=3, linear_nb_nonzero_elements_distribution=False),
        SSSApproximator(nb_states=50),
        SSSApproximator(nb_states=100),
        SSSApproximator(nb_states=200),
        SSSApproximator(nb_states=300),
    ]

    nb_param_shares = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    if not os.path.isdir(output_folderpath):
        os.mkdir(output_folderpath)

    log_filepath = os.path.join(output_folderpath, "approximation.log")
    write_header_to_file(title="APPROXIMATION LOG", log_filepath=log_filepath)

    y_pred_train = model.features_and_optim_mat_to_prediction(features=X_train_t, optim_mat=optim_mat_t)
    accuracy = get_accuracy(y_train_t, y_pred_train)
    log_to_file(txt="Original Train Accuracy: " + str(accuracy), log_filepath=log_filepath)
    y_pred_val = model.features_and_optim_mat_to_prediction(features=X_val_t, optim_mat=optim_mat_t)
    accuracy = get_accuracy(y_val_t, y_pred_val)
    log_to_file(txt="Original Val Accuracy: " + str(accuracy), log_filepath=log_filepath)

    for nb_params_share in nb_param_shares:
        for approximator in approximators:
            res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share)
            approximated_mat = res_dict["approx_mat_dense"]
            approximated_mat_t = torch.tensor(approximated_mat).float()

            y_pred_train = model.features_and_optim_mat_to_prediction(features=X_train_t, optim_mat=approximated_mat_t)
            accuracy = get_accuracy(y_train_t, y_pred_train)
            log_to_file(txt=approximator.get_name() + "_" + str(nb_params_share) + "nnz" + " Train Accuracy: " + str(accuracy), log_filepath=log_filepath)

            y_pred_val = model.features_and_optim_mat_to_prediction(features=X_val_t, optim_mat=approximated_mat_t)
            accuracy = get_accuracy(y_val_t, y_pred_val)
            log_to_file(txt=approximator.get_name() + "_" + str(nb_params_share) + "nnz" + " Val Accuracy: " + str(accuracy), log_filepath=log_filepath)
            
            output_filename = model_class.__name__ + "_" + label_tag + "_" + approximator.get_name() + "_" + str(nb_params_share) + "nnz" + "_res_dict.p"
            pickle.dump(res_dict, open(os.path.join(output_folderpath, output_filename), "wb"))

if __name__ == "__main__":
    model_class = InceptionV3
    features_filepath = "/path/to/InceptionV3_animal_features.p"
    output_folderpath = "/path/to/approximated_models/"
    label_filepath = get_animal_classes_filepath()

    benchmark_approximate_weight_matrix(
        model_class=model_class,
        features_filepath=features_filepath,
        output_folderpath=output_folderpath,
        label_filepath=label_filepath
    )