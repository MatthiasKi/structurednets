import os
import numpy as np

from structurednets.asset_helpers import get_validation_classes_filepath, get_animal_classes_filepath, get_object_classes_filepath

def print_imagenet_data_stats(label_filepath: str, training_folder_path: str):
    with open(get_validation_classes_filepath(), "r") as f:
        validation_labels = f.readlines()
    validation_labels = [label.split("\n")[0] for label in validation_labels]

    print(" ----------------------- ")
    print("")
    print("Stats for " + label_filepath)

    nb_train_images = []
    nb_val_images = []
    with open(label_filepath, "r") as f:
        labels = f.readlines()
        for label in labels:
            cat_id = label.split(" ")[0]
            cat_idx = label.split(" ")[1]
            nb_train_images.append(len(os.listdir(os.path.join(training_folder_path, cat_id))))
            nb_val_images.append(len([idx for idx in validation_labels if idx == cat_idx]))

    print("Number of categories: " + str(len(nb_train_images)))
    print("Mean Number of train images: " + str(np.mean(nb_train_images)) + "+-" + str(np.std(nb_train_images)))
    print("Mean Number of val images: " + str(np.mean(nb_val_images)) + "+-" + str(np.std(nb_val_images)))
    print("Min / Max number of train images: " + str(np.min(nb_train_images)) + " / " + str(np.max(nb_train_images)))
    print("Min / Max number of val images: " + str(np.min(nb_val_images)) + " / " + str(np.max(nb_val_images)))
    print("")

if __name__ == "__main__":
    training_folder_path = "/path/to/Imagenet/train/"
    label_filepaths = [
        get_animal_classes_filepath(),
        get_object_classes_filepath()
    ]

    for label_filepath in label_filepaths:
        print_imagenet_data_stats(label_filepath=label_filepath, training_folder_path=training_folder_path)