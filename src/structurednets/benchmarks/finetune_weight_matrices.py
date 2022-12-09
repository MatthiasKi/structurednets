import pickle
import torch
import os
import numpy as np

from structurednets.models.googlenet import GoogleNet
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.resnet18 import Resnet18
from structurednets.layers.hmat_layer import HMatLayer
from structurednets.layers.lr_layer import LRLayer
from structurednets.layers.psm_layer import PSMLayer
from structurednets.layers.sss_layer import SSSLayer
from structurednets.training_helpers import train_with_decreasing_lr, get_loss_and_accuracy_for_model, get_full_batch, get_train_data, transform_feature_dtypes
from structurednets.asset_helpers import load_features, get_all_classes_filepath, get_animal_classes_filepath
from structurednets.features.extract_features import get_required_indices, get_features_output_filename, get_inverse_class_map

def benchmark_finetune_weight_matrices(
    train_features_basepath: str,
    val_features_basepath: str,
    path_to_labelfile: str,
    results_filepath="weight_matrix_finetuning_result.p",
    pretrained_dicts_path="weight_matrices_approximation_result.p",
    patience=3,
    batch_size=1e9,
    min_patience_improvement=1e-6,
    loss_function_class=torch.nn.CrossEntropyLoss,
):
    weight_matrix_data = pickle.load(open(pretrained_dicts_path, "rb"))
    required_indices = get_required_indices(path_to_label_file=path_to_labelfile)

    model_classes = [
        GoogleNet,
        InceptionV3,
        MobilenetV2,
        Resnet18,
    ]

    result = dict()
    for model_class in model_classes:
        train_features_path = os.path.join(train_features_basepath, get_features_output_filename(model_class, path_to_labelfile))
        val_features_path = os.path.join(val_features_basepath, get_features_output_filename(model_class, path_to_labelfile))

        model_name = model_class.__name__
        model = model_class(output_indices=required_indices, use_gpu=False)
        weight_matrix = model.get_optimization_matrix().detach().numpy().T
        input_dim = weight_matrix.shape[1]
        output_dim = weight_matrix.shape[0]
        nb_params_share = 0.2

        print("------------------ Starting " + model_name + " ---------------------")

        X_train, X_val, y_train, y_val = get_train_data(train_features_path)
        # NOTE: We report the results on the original Imagenet validation dataset (which is not used during our training period). To avoid confusion, we name this dataset "test set" here. It is a test set for us - but not for the originally trained models. 
        X_test, y_test, _ = load_features(val_features_path)

        # NOTE: We are here working on the outputs *before* they get class-adapted - hence we need to re-adjust the labels which have been extracted
        inverse_class_map = get_inverse_class_map(labels_filepath=path_to_labelfile)
        y_train = np.array([inverse_class_map[entry] for entry in y_train])
        y_val = np.array([inverse_class_map[entry] for entry in y_val])
        y_test = np.array([inverse_class_map[entry] for entry in y_test])

        X_train, X_val, y_train, y_val = transform_feature_dtypes(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
        X_test, _, y_test, _ = transform_feature_dtypes(X_train=X_test, X_val=X_test, y_train=y_test, y_val=y_test)

        X_test_t, y_test_t = get_full_batch(X=X_test, y=y_test)

        approximator_names = [
            #'HMatApproximatorWrapper', 
            'LRApproximator',
            # 'PSMApproximatorWrapper', 
            # 'SSSApproximatorWrapper'
        ]
        layers = [
            # HMatLayer(
            #     input_dim=input_dim,
            #     output_dim=output_dim,
            #     nb_params_share=nb_params_share,
            #     initial_hmatrix=weight_matrix_data[nb_params_share]["HMatApproximatorWrapper"][model_name]["res_dict"]["h_matrix"]
            # ),
            LRLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                nb_params_share=nb_params_share,
                initial_lr_components=[weight_matrix_data[nb_params_share]["LRApproximator"][model_name]["res_dict"]["left_mat"], weight_matrix_data[0.2]["LRApproximator"][model_name]["res_dict"]["right_mat"]]
            ),
            # PSMLayer(
            #     input_dim=input_dim,
            #     output_dim=output_dim,
            #     nb_params_share=nb_params_share,
            #     sparse_matrices=weight_matrix_data[nb_params_share]["PSMApproximatorWrapper"][model_name]["res_dict"]["faust_approximation"]
            # ),
            # SSSLayer(
            #     input_dim=input_dim,
            #     output_dim=output_dim,
            #     nb_params_share=nb_params_share,
            #     initial_system_approx=weight_matrix_data[nb_params_share]["SSSApproximatorWrapper"][model_name]["res_dict"]["system_approx"],
            #     nb_states=weight_matrix_data[nb_params_share]["SSSApproximatorWrapper"][model_name]["res_dict"]["nb_states"]
            # )
        ]
        
        for approximator_name, layer in zip(approximator_names, layers):
            print("Now checking " + approximator_name)
            test_loss_before_training, test_accuracy_before_training = get_loss_and_accuracy_for_model(model=layer, X_t=X_test_t, y_t=y_test_t, loss_function_class=loss_function_class)
            trained_layer, _, _, _, _, train_loss_histories, train_accuracy_histories, val_loss_histories, val_accuracy_histories = train_with_decreasing_lr(model=layer, X_train=X_train, y_train=y_train, X_val=None, y_val=None, patience=patience, batch_size=batch_size, verbose=True, loss_function_class=loss_function_class, min_patience_improvement=min_patience_improvement, optimizer_class=torch.optim.SGD)
            test_loss_after_training, test_accuracy_after_training = get_loss_and_accuracy_for_model(model=trained_layer, X_t=X_test_t, y_t=y_test_t, loss_function_class=loss_function_class)

            result[approximator_name] = dict()
            result[approximator_name]["test_accuracy_before_training"] = test_accuracy_before_training
            result[approximator_name]["test_loss_before_training"] = test_loss_before_training
            result[approximator_name]["test_accuracy_after_training"] = test_accuracy_after_training
            result[approximator_name]["test_loss_after_training"] = test_loss_after_training
            result[approximator_name]["trained_layer"] = trained_layer
            result[approximator_name]["train_loss_histories"] = train_loss_histories
            result[approximator_name]["train_accuracy_histories"] = train_accuracy_histories
            result[approximator_name]["val_loss_histories"] = val_loss_histories
            result[approximator_name]["val_accuracy_histories"] = val_accuracy_histories

            print("Trained for " + str(sum([len(l) for l in train_loss_histories])) + " steps")

            pickle.dump(result, open(results_filepath, "wb"))

if __name__ == "__main__":
    train_features_basepath = "/home/ga76sih/lrz-nashome/Imagenet/features/" # "path/to/train_features/"
    val_features_basepath = "/home/ga76sih/lrz-nashome/Imagenet/validation_features/" # "path/to/val_features/"
    pretrained_dicts_path = "/home/ga76sih/lrz-nashome/structuresurvey/with_hedlr_3/weight_matrices_approximation_result.p" # "weight_matrices_approximation_result.p"
    results_filepath = "/home/ga76sih/lrz-nashome/structuresurvey/with_hedlr_3/weight_matrix_finetuning_result_fred01_lr.p" # weight_matrix_finetuning_result.p
    path_to_labelfile = get_all_classes_filepath()

    benchmark_finetune_weight_matrices(
        train_features_basepath=train_features_basepath,
        val_features_basepath=val_features_basepath,
        path_to_labelfile=path_to_labelfile,
        pretrained_dicts_path=pretrained_dicts_path,
        results_filepath=results_filepath,
    )