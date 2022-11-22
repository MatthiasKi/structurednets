import os
import importlib_resources
import pickle
import numpy as np
import scipy.io

def get_label_folder_path() -> str:
    return os.path.join(importlib_resources.files("structurednets"), "labels")

def get_test_matrices_folder_path() -> str:
    return os.path.join(importlib_resources.files("structurednets"), "testmatrices")

def get_all_classes_filepath() -> str:
    return os.path.join(get_label_folder_path(), "all_imagenet_classes.txt")

def get_animal_classes_filepath() -> str:
    return os.path.join(get_label_folder_path(), "animal_imagenet_classes.txt")

def get_object_classes_filepath() -> str:
    return os.path.join(get_label_folder_path(), "object_imagenet_classes.txt")

def get_validation_classes_filepath() -> str:
    return os.path.join(get_label_folder_path(), "ILSVRC2012_validation_ground_truth.txt")

def get_label_name_from_path(label_filepath: str) -> str:
    return label_filepath.split("/")[-1].split("_")[0]

def get_validation_metadata_filepath() -> str:
    return os.path.join(get_label_folder_path(), "meta.mat")

def assemble_features_output_filename(model_name: str, labels_tag: str):
    return model_name + "_" + labels_tag + "_features.p"

def get_all_test_matrix_dicts() -> list:
    folder_path = get_test_matrices_folder_path()
    test_matrices_filenames = os.listdir(folder_path)
    test_matrices_filenames = [filename for filename in test_matrices_filenames if filename[-2:] == ".p"]
    test_matrix_dicts = [pickle.load(open(os.path.join(folder_path, filename), "rb")) for filename in test_matrices_filenames]
    return test_matrix_dicts

def load_features(path_to_feature_file: str):
    X, y, y_pred = pickle.load(open(path_to_feature_file, "rb"))
    X = np.squeeze(X)
    y = np.squeeze(y)
    y_pred = np.squeeze(y_pred)
    return X, y, y_pred

def get_output_idx_for_validation_imagename(imagename: str, label_filepath: str) -> int:
    classname = get_classname_for_validation_imagename(imagename)
    output_idx = get_output_idx_for_classname(classname, label_filepath)
    return output_idx

def get_classname_for_validation_imagename(imagename: str) -> str:
    metadata_file = scipy.io.loadmat(get_validation_metadata_filepath())["synsets"]

    line_number = int(imagename.split("_")[-1].split(".")[0]) - 1

    with open(get_validation_classes_filepath(), "r") as groundtruth_file:
        line = groundtruth_file.readline()
        for _ in range(line_number):
            line = groundtruth_file.readline()

        class_number = int(line.split("\n")[0])

        metadata = metadata_file[class_number-1]
        metadata_class_number = metadata[0][0][0][0]
        assert metadata_class_number == class_number, "The class number found in the meta data does not match the targeted class number"
        classname = metadata[0][1][0]

        return classname

def get_output_idx_for_classname(classname: str, label_filepath: str) -> int:
    with open(label_filepath, "r") as labelfile:
        line = labelfile.readline()
        idx = 0
        while line:
            line_classname = line.split(" ")[0]
            if line_classname == classname:
                return idx
            line = labelfile.readline()
            idx += 1

    return -1