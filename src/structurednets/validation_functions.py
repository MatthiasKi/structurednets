import pickle
import os
import torch
import numpy as np

from structurednets.asset_helpers import get_validation_metadata_filepath, get_validation_classes_filepath, get_animal_classes_filepath, assemble_features_output_filename, load_features
from structurednets.training_helpers import get_accuracy
from structurednets.layers.psmlayer import build_PSMLayer_from_res_dict
from structurednets.models.model_helpers import visionmodel_name_to_class
from structurednets.features.extract_features import get_required_indices

def compute_original_model_accuracies(res_csv_path: str, features_folderpath: str):
    with open(res_csv_path, "w") as csv_file:
        csv_file.write(f"ModelName,CategoryTag,Accuracy\n")

        feature_filenames = os.listdir(features_folderpath)
        for feature_filename in feature_filenames:
            feature_filepath = os.path.join(features_folderpath, feature_filename)

            _, y, y_pred = load_features(path_to_feature_file=feature_filepath)
            accuracy = float(np.sum(y == y_pred) / len(y))

            model_name = feature_filename.split("_")[0]
            category_tag = feature_filename.split("_")[1]

            csv_file.write(f"{model_name},{category_tag},{accuracy}\n")

def compute_model_accuracy(resdict_path: str, features_filepath: str):
    # NOTE that this function does only work with resdicts produced by sss_model_training.py or standard_model_training.py
    X, y, _ = load_features(path_to_feature_file=features_filepath)
    y_t = torch.tensor(y)
    res_dict = pickle.load(open(resdict_path, "rb"))
    model = res_dict["model"]
    X_t = torch.tensor(X)
    pred = model(X_t)
    return get_accuracy(y_t, pred)

def compute_psm_layer_accuracy(resdict_path: str, features_filepath: str, psm_layer_bias: np.ndarray):
    X, y, _ = load_features(path_to_feature_file=features_filepath)
    y_t = torch.tensor(y)
    res_dict = pickle.load(open(resdict_path, "rb"))
    model = build_PSMLayer_from_res_dict(res_dict=res_dict, bias=psm_layer_bias)
    X_t = torch.tensor(X)
    pred = model(X_t)
    return get_accuracy(y_t, pred)

def check_psm_layer_parameter_count(resdict_path: str, psm_layer_bias: np.ndarray, param_share: float) -> bool:
    res_dict = pickle.load(open(resdict_path, "rb"))
    model = build_PSMLayer_from_res_dict(res_dict=res_dict, bias=psm_layer_bias)
    return model.get_nb_parameters_in_weight_matrix() <= int(res_dict["approx_mat_dense"].size * param_share)

def get_psmapproximator_param_share_filenames(filenames: list, param_share: float) -> list:
    res = []
    for filename in filenames:
        if (str(param_share) + "nnz") in filename:
            res.append(filename)
    return res

def get_validation_accuracy_from_psm_approximation_log(folder_path: str, filename: str) -> float:
    with open(os.path.join(folder_path, "approximation.log"), "r") as log:
        line = log.readline()
        while line:
            contains_all_parts = True
            parts = filename.split("_")[2:-2]
            for part in parts:
                if part not in line:
                    contains_all_parts = False
                    break
            if contains_all_parts and "Val Accuracy" in line:
                return float(line.split(":")[-1].split(" ")[-1])
            line = log.readline()

    raise Exception("Could not find " + filename + " in the psm layer approximation log")

if __name__ == "__main__":
    # Check if the output idx is extracted correctly
    # test_imagename = "ILSVRC2012_val_00050000.JPEG"
    # classname = get_classname_for_validation_imagename(test_imagename)
    # output_idx = get_output_idx_for_classname(classname, get_animal_classes_filepath())
    # ---

    # Compute the validation accuracies for the speed experiments
    # run_name = "speed_experiments_2"
    # validation_models_folder = f"/path/to/{run_name}"
    # output_filepath = f"/path/to/validation_results_{run_name}.csv"
    # features_basefolder = "/path/to/validation_features"

    # with open(output_filepath, "w") as output_file:
    #     output_file.write("Filename,Accuracy\n")

    #     filenames = os.listdir(validation_models_folder)
    #     for filename in filenames:
    #         if not (".log" in filename):
    #             res_dict_path = os.path.join(validation_models_folder, filename)

    #             model_name = filename.split("_")[0]
    #             labelfile_type = filename.split("_")[1]
    #             features_filename = assemble_features_output_filename(model_name, labelfile_type)
    #             features_filepath = os.path.join(features_basefolder, features_filename)

    #             accuracy = compute_model_accuracy(res_dict_path, features_filepath)
    #             output_file.write(f"{filename},{accuracy}\n")
    # ---

    # Compute the validation accuracies for the original models
    # res_csv_path = "/path/to/original_model_validation_accuracies.csv"
    # features_folderpath = "/path/to/validation_features/"
    # compute_original_model_accuracies(res_csv_path, features_folderpath)
    # ---

    # Compute the validation accuracies for PSM Models
    run_name = "structure_choice_benchmark"
    validation_models_folder = f"/path/to/{run_name}"
    output_filepath = f"/path/to/validation_results_{run_name}.csv"
    features_basefolder = "/path/to/validation_features"
    model_name = "InceptionV3"
    label_filepath = get_animal_classes_filepath()
    labelfile_type = "animal"

    required_indices = get_required_indices(label_filepath)
    original_model = visionmodel_name_to_class(model_name)(required_indices)
    psm_layer_bias = original_model.get_bias()
    
    features_filename = assemble_features_output_filename(model_name, labelfile_type)
    features_filepath = os.path.join(features_basefolder, features_filename)

    with open(output_filepath, "w") as output_file:
        output_file.write("Filename,Accuracy\n")

        filenames = os.listdir(validation_models_folder)
        nb_param_shares = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        for param_share in nb_param_shares:
            param_share_filenames = get_psmapproximator_param_share_filenames(filenames, param_share)
            assert len(param_share_filenames) == 4, f"There should be exactly 4 res dicts for each param share (found {len(param_share_filenames)} for {param_share})"

            best_param_share_filename = None
            best_param_share_accuracy = None
            for param_share_filename in param_share_filenames:
                curr_param_share_accuracy = get_validation_accuracy_from_psm_approximation_log(validation_models_folder, param_share_filename)
                if best_param_share_accuracy is None or curr_param_share_accuracy > best_param_share_accuracy:
                    best_param_share_accuracy = curr_param_share_accuracy
                    best_param_share_filename = param_share_filename

            res_dict_path = os.path.join(validation_models_folder, best_param_share_filename)
            assert check_psm_layer_parameter_count(res_dict_path, psm_layer_bias, param_share), "Too many parameters in the PSM layer"
            accuracy = compute_psm_layer_accuracy(res_dict_path, features_filepath, psm_layer_bias)
            output_file.write(f"{best_param_share_filename},{accuracy}\n")
    # ---
