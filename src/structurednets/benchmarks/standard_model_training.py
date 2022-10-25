import torch
import os
import pickle
import numpy as np

from structurednets.logging_helpers import write_header_to_file
from structurednets.asset_helpers import get_animal_classes_filepath
from structurednets.models.visionmodel import VisionModel
from structurednets.models.alexnet import AlexNet
from structurednets.features.extract_features import get_required_indices
from structurednets.training_helpers import train_with_features

def benchmark_train_standard_model(features_filepath: str, path_to_labelfile: str, model_class: VisionModel, patience: int, batch_sizes: list, verbose_train_progress: bool, lrs: list, train_from_scratch: bool, use_softmaxes: list, output_foldername: str):
    if not os.path.isdir(output_foldername):
        os.mkdir(output_foldername)

    log_filepath = os.path.join(output_foldername, "standard_training.log")
    write_header_to_file(title="STANDARD MODEL TRAINING", log_filepath=log_filepath)

    required_indices = get_required_indices(path_to_label_file=path_to_labelfile)
    orig_model = model_class(output_indices=required_indices)

    start_weight_mat = orig_model.get_optimization_matrix()
    start_bias = orig_model.get_bias()

    for use_softmax in use_softmaxes:
        for lr in lrs:
            for batch_size in batch_sizes:
                linear_layer = torch.nn.Linear(in_features=start_weight_mat.shape[0], out_features=start_weight_mat.shape[1], bias=True)
                if not train_from_scratch:
                    linear_layer.weight = torch.nn.Parameter(start_weight_mat.T)
                    linear_layer.bias = torch.nn.Parameter(start_bias)
                if use_softmax:
                    model = torch.nn.Sequential(
                        linear_layer,
                        torch.nn.Softmax(),
                    )
                else:
                    model = torch.nn.Sequential(
                        linear_layer,
                    )

                start_train_loss, start_train_accuracy, start_val_loss, start_val_accuracy, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train_with_features(model=model, features_path=features_filepath, patience=patience, batch_size=batch_size, verbose=verbose_train_progress, lr=lr)

                output_filename = model_class.__name__ + "_standard_finetune_res_lr_" + str(lr) + "_batch_size_" + str(batch_size)
                if train_from_scratch:
                    output_filename += "_from_scratch"
                if not use_softmax:
                    output_filename += "_wo_softmax"
                output_filename += ".p"

                res = dict()
                res["start_train_loss"] = start_train_loss
                res["start_train_accuracy"] = start_train_accuracy
                res["start_val_loss"] = start_val_loss
                res["start_val_accuracy"] = start_val_accuracy
                res["train_loss_history"] = train_loss_history
                res["train_accuracy_history"] = train_accuracy_history
                res["val_loss_history"] = val_loss_history
                res["val_accuracy_history"] = val_accuracy_history
                res["model"] = model
                pickle.dump(res, open(os.path.join(output_foldername, output_filename), "wb"))

                with open(log_filepath, "a") as f:
                    f.write("#" * 20 + "\n")
                    f.write("Use Softmax: " + str(use_softmax) + "\n")
                    f.write("Learning Rate: " + str(lr) + "\n")
                    f.write("Batch Size: " + str(batch_size) + "\n")
                    f.write("Best Train Accuracy: " + str(np.max(train_accuracy_history)) + "\n")
                    f.write("Best Val Accuracy: " + str(np.max(val_accuracy_history)) + "\n")
                    f.write("#" * 20 + "\n")

if __name__ == "__main__":
    features_filepath = "/path/to/AlexNet_animal_features.p"
    path_to_labelfile = get_animal_classes_filepath()
    model_class = AlexNet
    patience = 10
    batch_sizes = [50, 500, 1000, 5000]
    verbose_train_progress = False
    lrs = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    train_from_scratch = True
    use_softmaxes = [True, False]
    output_foldername = "/path/to/finetuned_models/"

    benchmark_train_standard_model(
        features_filepath=features_filepath,
        path_to_labelfile=path_to_labelfile,
        model_class=model_class,
        patience=patience,
        batch_sizes=batch_sizes,
        verbose_train_progress=verbose_train_progress,
        lrs=lrs,
        train_from_scratch=train_from_scratch,
        use_softmaxes=use_softmaxes,
        output_foldername=output_foldername
    )