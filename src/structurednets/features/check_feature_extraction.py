import pickle
import numpy as np
import torch

from structurednets.asset_helpers import get_animal_classes_filepath, get_object_classes_filepath, load_features, get_all_classes_filepath
from structurednets.features.extract_features import get_required_indices
from structurednets.models.visionmodel import VisionModel
from structurednets.models.alexnet import AlexNet
from structurednets.models.googlenet import GoogleNet
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.resnet18 import Resnet18
from structurednets.models.vgg16 import VGG16

def check_features(feature_filepath: str, model_class: VisionModel, label_filepath: str):
    required_indices = get_required_indices(label_filepath)
    model = model_class(required_indices, use_gpu=False)
    output_mat = model.get_optimization_matrix()

    X, y, y_pred = load_features(path_to_feature_file=feature_filepath)

    y_featured_pred = model.features_and_optim_mat_to_prediction_with_argmax(torch.tensor(X), output_mat).cpu().detach().numpy()
    assert np.array_equal(y_pred, y_featured_pred), "Features times output matrix do not match the predictions"

    accuracies = []
    y_values = np.unique(y)
    total_nb_correct_predictions = 0
    for y_value in y_values:
        mask = y == y_value
        curr_ground_truth = y[mask]
        curr_predictions = y_pred[mask]
        accuracies.append(float(np.sum(curr_ground_truth == curr_predictions)) / np.sum(mask))
        total_nb_correct_predictions += np.sum(curr_ground_truth == curr_predictions)
        print("Accuracy for class " + str(y_value) + ": " + str(accuracies[-1]))
    total_accuracy = float(total_nb_correct_predictions) / len(y)

    print("---------------")
    print("Min accuracy: " + str(np.min(accuracies)))
    print("Max accuracy: " + str(np.max(accuracies)))
    print("Mean Accuracy: " + str(np.mean(accuracies)) + "+-" + str(np.std(accuracies)))
    print("Median Accuracy: " + str(np.median(accuracies)))
    print("Total Accuracy: " + str(total_accuracy))

if __name__ == "__main__":
    feature_filepath = "/path/to/AlexNet_animal_features.p"
    model_class = AlexNet
    label_filepath = get_all_classes_filepath()

    check_features(feature_filepath=feature_filepath, model_class=model_class, label_filepath=label_filepath)