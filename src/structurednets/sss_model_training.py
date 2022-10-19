import os
import pickle
import numpy as np
import torch

from structurednets.logging_helpers import write_header_to_file
from structurednets.asset_helpers import get_animal_classes_filepath, get_object_classes_filepath, get_label_name_from_path
from structurednets.models.alexnet import AlexNet
from structurednets.models.googlenet import GoogleNet
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.resnet18 import Resnet18
from structurednets.models.vgg16 import VGG16
from structurednets.models.visionmodel import VisionModel
from structurednets.features.extract_features import get_required_indices, get_features_output_filename
from structurednets.training_helpers import train
from structurednets.layers.sss_layer import SemiseparableLayer

def train_sss_model(
    features_basepath: str, 
    paths_to_labelfile: list, 
    model_classes: list, 
    patience: int, 
    batch_sizes: list, 
    verbose_train_progress: bool, 
    lrs: list, 
    use_softmaxes: list, 
    output_foldername: str, 
    nb_param_shares: list,
    start_from_scratches: list,
    nb_states_list: list,
    iterations=1,
):
    if not os.path.isdir(output_foldername):
        os.mkdir(output_foldername)

    log_filepath = os.path.join(output_foldername, "sss_training.log")
    write_header_to_file(title="SSS MODEL TRAINING", log_filepath=log_filepath)

    for model_class in model_classes:
        for path_to_labelfile in paths_to_labelfile:
            required_indices = get_required_indices(path_to_label_file=path_to_labelfile)
            orig_model = model_class(output_indices=required_indices)

            features_filepath = os.path.join(features_basepath, get_features_output_filename(model_class, path_to_labelfile))

            start_weight_mat = orig_model.get_optimization_matrix()
            start_bias = orig_model.get_bias()

            input_size = start_weight_mat.shape[0]
            output_size = start_weight_mat.shape[1]

            # TODO actually I am not really sure if this *really* is a problem?! states could stil have 0 dimensions, and then it should be fine?!
            assert min(output_size, input_size) >= min(nb_states_list), "Nb_states can not be larger than the minimum input / output dim"

            for iteration in range(iterations):
                for nb_param_share in nb_param_shares:
                    for start_from_scratch in start_from_scratches:
                        for nb_states in nb_states_list:
                            for use_softmax in use_softmaxes:
                                for lr in lrs:
                                    for batch_size in batch_sizes:
                                        if start_from_scratch:
                                            initial_T = None
                                            initial_bias = None
                                        else:
                                            initial_T = start_weight_mat.cpu().detach().numpy().T
                                            initial_bias = start_bias
                                        model = SemiseparableLayer(input_size=input_size, output_size=output_size, nb_params_share=nb_param_share, nb_states=nb_states, initial_T=initial_T, initial_bias=initial_bias)

                                        if use_softmax:
                                            model = torch.nn.Sequential(model, torch.nn.Softmax())

                                        label_name = get_label_name_from_path(path_to_labelfile)
                                        output_filename = model_class.__name__ + "_" + label_name + "_sss_finetune_res_lr_" + str(lr) + "_batch_size_" + str(batch_size) + "_paramshare_" + str(nb_param_share) + "_states_" + str(nb_states)
                                        if not use_softmax:
                                            output_filename += "_wo_softmax"
                                        if start_from_scratch:
                                            output_filename +="_from_scratch"
                                        output_filename += "_iter_" + str(iteration) + ".p"

                                        res = dict()
                                        res["model"] = model
                                        res["nb_param_share"] = nb_param_share
                                        res["nb_states"] = nb_states
                                        res["start_from_scratch"] = start_from_scratch
                                        res["label_name"] = label_name

                                        if not start_from_scratch:
                                            only_approx_output_filename = output_filename[:-2]
                                            only_approx_output_filename += "_only_approx.p"
                                            pickle.dump(res, open(os.path.join(output_foldername, only_approx_output_filename), "wb"))

                                        model, start_train_loss, start_train_accuracy, start_val_loss, start_val_accuracy, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train(model=model, features_path=features_filepath, patience=patience, batch_size=batch_size, verbose=verbose_train_progress, lr=lr, restore_best_model=True)

                                        res["start_train_loss"] = start_train_loss
                                        res["start_train_accuracy"] = start_train_accuracy
                                        res["start_val_loss"] = start_val_loss
                                        res["start_val_accuracy"] = start_val_accuracy
                                        res["train_loss_history"] = train_loss_history
                                        res["train_accuracy_history"] = train_accuracy_history
                                        res["val_loss_history"] = val_loss_history
                                        res["val_accuracy_history"] = val_accuracy_history
                                        pickle.dump(res, open(os.path.join(output_foldername, output_filename), "wb"))
                                        with open(log_filepath, "a") as f:
                                            f.write("#" * 20 + "\n")
                                            f.write("Use Softmax: " + str(use_softmax) + "\n")
                                            f.write("Learning Rate: " + str(lr) + "\n")
                                            f.write("Batch Size: " + str(batch_size) + "\n")
                                            f.write("Nb Param Share: " + str(nb_param_share) + "\n")
                                            f.write("Nb States: " + str(nb_states) + "\n")
                                            f.write("Start from Scratch: " + str(start_from_scratch) + "\n")
                                            f.write("Best Train Accuracy: " + str(np.max(train_accuracy_history)) + "\n")
                                            f.write("Best Val Accuracy: " + str(np.max(val_accuracy_history)) + "\n")
                                            f.write("#" * 20 + "\n")

if __name__ == "__main__": 
    features_basepath = "/path/to/features/"
    paths_to_labelfile = [get_animal_classes_filepath()]
    model_classes = [InceptionV3]
    patience = 10
    batch_sizes = [64]
    verbose_train_progress = True
    lrs = [1e-3]
    nb_param_shares = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] # [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    use_softmaxes = [True]
    start_from_scratches = [False]
    nb_states_list = [50]
    output_foldername = "/path/to/outputfolder/"
    iterations = 5

    train_sss_model(
        features_basepath=features_basepath, 
        paths_to_labelfile=paths_to_labelfile,
        model_classes=model_classes, 
        patience=patience, 
        batch_sizes=batch_sizes, 
        verbose_train_progress=verbose_train_progress,
        lrs=lrs,
        use_softmaxes=use_softmaxes,
        output_foldername=output_foldername,
        nb_param_shares=nb_param_shares,
        start_from_scratches=start_from_scratches,
        nb_states_list=nb_states_list,
        iterations=iterations
    )