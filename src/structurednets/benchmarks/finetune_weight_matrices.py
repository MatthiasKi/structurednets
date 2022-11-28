import numpy as np
import pickle
import torch

from structurednets.models.googlenet import GoogleNet
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.resnet18 import Resnet18
from structurednets.layers.hmat_layer import HMatLayer
from structurednets.layers.lr_layer import LRLayer
from structurednets.layers.psm_layer import PSMLayer
from structurednets.layers.sss_layer import SSSLayer
from structurednets.training_helpers import train_with_decreasing_lr, get_loss_and_accuracy_for_model, get_full_batch, get_train_data, transform_feature_dtypes
from structurednets.asset_helpers import load_features

def benchmark_finetune_weight_matrices(
    train_features_path: str,
    val_features_path: str,
    results_filepath="weight_matrix_finetuning_result.p",
    pretrained_dicts_path="test_matrices_approximation_result.p",
    patience=10,
    batch_size=1000,
    min_patience_improvement=1e-6,
    loss_function_class=torch.nn.CrossEntropyLoss
):
    weight_matrix_data = pickle.load(open(pretrained_dicts_path, "rb"))
    required_indices = np.arange(1000)

    model_classes = [
        GoogleNet,
        InceptionV3,
        MobilenetV2,
        Resnet18,
    ]

    result = dict()
    for model_class in model_classes:
        model_name = model_class.__name__
        model = model_class(output_indices=required_indices, use_gpu=False)
        weight_matrix = model.get_optimization_matrix().detach().numpy()
        input_dim = weight_matrix.shape[1]
        output_dim = weight_matrix.shape[0]

        X_train, X_val, y_train, y_val = get_train_data(train_features_path)
        X_train, X_val, y_train, y_val = transform_feature_dtypes(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

        # NOTE: We report the results on the original Imagenet validation dataset (which is not used during our training period). To avoid confusion, we name this dataset "test set" here. It is a test set for us - but not for the originally trained models. 
        X_test, y_test, _ = load_features(val_features_path)
        X_test, _, y_test, _ = transform_feature_dtypes(X_train=X_test, X_val=X_test, y_train=y_test, y_val=y_test)
        X_test_t, y_test_t = get_full_batch(X=X_test, y=y_test)

        approximator_names = [
            'HMatApproximatorWrapper', 
            'LRApproximator', 
            'PSMApproximatorWrapper', 
            'SSSApproximatorWrapper'
        ]
        layers = [
            HMatLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                initial_hmatrix=weight_matrix_data[0.2]["HMatApproximatorWrapper"][model_name]["res_dict"]["h_matrix"]
            ),
            LRLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                initial_lr_components=[weight_matrix_data[0.2]["LRApproximator"][model_name]["res_dict"]["left_mat"], weight_matrix_data[0.2]["LRApproximator"][model_name]["res_dict"]["right_mat"]]
            ),
            PSMLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                sparse_matrices=weight_matrix_data[0.2]["PSMApproximatorWrapper"][model_name]["res_dict"]["faust_approximation"]
            ),
            SSSLayer(
                input_dim=input_dim,
                output_dim=output_dim,
                initial_system_approx=weight_matrix_data[0.2]["SSSApproximatorWrapper"][model_name]["res_dict"]["system_approx"]

            )
        ]
        
        for approximator_name, layer in zip(approximator_names, layers):
            test_loss_before_training, test_accuracy_before_training = get_loss_and_accuracy_for_model(model=layer, X_t=X_test_t, y_t=y_test_t, loss_function_class=loss_function_class)
            trained_layer, _, _, _, _, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train_with_decreasing_lr(model=layer, X_train=X_train, y_train=y_train, X_val=None, y_val=None, patience=patience, batch_size=batch_size, verbose=False, loss_function_class=loss_function_class, min_patience_improvement=min_patience_improvement, optimizer_class=torch.optim.SGD, finetune_mode=True)
            test_loss_after_training, test_accuracy_after_training = get_loss_and_accuracy_for_model(model=trained_layer, X_t=X_test_t, y_t=y_test_t, loss_function_class=loss_function_class)

            result[approximator_name] = dict()
            result[approximator_name]["test_accuracy_before_training"] = test_accuracy_before_training
            result[approximator_name]["test_loss_before_training"] = test_loss_before_training
            result[approximator_name]["test_accuracy_after_training"] = test_accuracy_after_training
            result[approximator_name]["test_loss_after_training"] = test_loss_after_training
            result[approximator_name]["trained_layer"] = trained_layer
            result[approximator_name]["train_loss_history"] = train_loss_history
            result[approximator_name]["train_accuracy_history"] = train_accuracy_history
            result[approximator_name]["val_loss_history"] = val_loss_history
            result[approximator_name]["val_accuracy_history"] = val_accuracy_history

        pickle.dump(result, open(results_filepath, "wb"))

if __name__ == "__main__":
    train_features_path = "path/to/train_features.p"
    val_features_path = "path/to/val_features.p"

    benchmark_finetune_weight_matrices(
        train_features_path=train_features_path,
        val_features_path=val_features_path,
    )