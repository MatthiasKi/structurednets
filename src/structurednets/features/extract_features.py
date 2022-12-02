import torch
import os
import numpy as np
import pickle
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image

from structurednets.asset_helpers import get_animal_classes_filepath, get_object_classes_filepath, get_label_name_from_path, assemble_features_output_filename, get_output_idx_for_validation_imagename, get_all_classes_filepath
from structurednets.models.visionmodel import VisionModel
from structurednets.models.alexnet import AlexNet
from structurednets.models.googlenet import GoogleNet
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.resnet18 import Resnet18
from structurednets.models.vgg16 import VGG16

def get_device(use_gpu=True):
    if use_gpu:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return 'cpu'

def get_label_file_lines(path_to_label_file: str):
    with open(path_to_label_file, "r") as f:
        label_lines = f.readlines()
        return label_lines

def get_required_indices(path_to_label_file: str):
    label_lines = get_label_file_lines(path_to_label_file=path_to_label_file)
    # NOTE The correct mapping from network output index to class is given by: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    required_indices = []
    for label_line in label_lines:
        required_indices.append(int(label_line.split(" ")[3]))
    # NOTE it is important that the required indices are NOT sorted, otherwise the ordering with the y labels get corrupted

    return required_indices

def get_features_output_filename(model_class: VisionModel, label_filepath: str):
    return assemble_features_output_filename(model_class.__name__, get_label_name_from_path(label_filepath))

def get_preprocessing_transformation():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256), # NOTE This ensures that the smaller dimension has at least 256 pixels
            normalize,
    ])
    return img_transformation

def extract_validation_features(output_folder_path: str, validation_folder_path: str, model_class: VisionModel, label_filepath: str, use_gpu=True):
    device = get_device(use_gpu=use_gpu)
    output_filename = get_features_output_filename(model_class, label_filepath)

    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    required_indices = get_required_indices(label_filepath)

    model = model_class(output_indices=required_indices)
    img_transformation = get_preprocessing_transformation()

    X = []
    y = []
    y_pred = []

    image_filenames = os.listdir(validation_folder_path)
    image_filenames = [img for img in image_filenames if ".JPEG" in img]
    for image_filename in image_filenames:
        output_idx = get_output_idx_for_validation_imagename(image_filename, label_filepath)
        if output_idx >= 0:
            img = Image.open(os.path.join(validation_folder_path, image_filename)).convert("RGB")
            tensor_img = img_transformation(img)
            batched_img = torch.unsqueeze(tensor_img, 0)
            batched_img = batched_img.to(device)

            X.append(model.get_features_for_batch(batched_img).cpu().detach().numpy())
            y.append(output_idx)
            y_pred.append(model.predict_pretrained_with_argmax(batched_img).cpu().detach().numpy())

    X = np.array(X)
    y = np.array(y)
    y_pred = np.array(y_pred)

    pickle.dump([X, y, y_pred], open(os.path.join(output_folder_path, output_filename), "wb"))

def extract_features(output_folder_path: str, training_folder_path: str, model_class: VisionModel, label_filepath: str, use_gpu=True):
    output_filename = get_features_output_filename(model_class, label_filepath)
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

    X = None
    y = None
    y_pred = None

    label_lines = get_label_file_lines(path_to_label_file=label_filepath)
    for line_i, line in enumerate(label_lines):
        folder_name = line.split(" ")[0]
        image_names = os.listdir(os.path.join(training_folder_path, folder_name))
        image_paths = [os.path.join(training_folder_path, folder_name, image_name) for image_name in image_names]

        X_append, y_append, y_pred_append = extract_images(
            image_paths=image_paths,
            model_class=model_class,
            label_filepath=label_filepath,
            label=line_i,
            use_gpu=use_gpu
        )

        if X is None:
            X = X_append
            y = y_append
            y_pred = y_pred_append
        else:
            X = np.concatenate([X, X_append], axis=0)
            y = np.concatenate([y, y_append], axis=0)
            y_pred = np.concatenate([y_pred, y_pred_append], axis=0)

    pickle.dump([X, y, y_pred], open(os.path.join(output_folder_path, output_filename), "wb"))

def extract_images(
    image_paths: list,
    model_class: VisionModel, 
    label_filepath: str,
    label: int,
    use_gpu=True,
):
    device = get_device(use_gpu=use_gpu)
    img_transformation = get_preprocessing_transformation()
    required_indices = get_required_indices(label_filepath)
    model = model_class(output_indices=required_indices)

    X = []
    y = []
    y_pred = []

    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        tensor_img = img_transformation(img)
        batched_img = torch.unsqueeze(tensor_img, 0)
        batched_img = batched_img.to(device)

        # NOTE: We do not put the images into one batch together (which would make the computation much faster), because
        # the images can have different sizes (I trade better features for slower computation time here)

        X.append(model.get_features_for_batch(batched_img).cpu().detach().numpy())
        y.append(label)
        y_pred.append(model.predict_pretrained_with_argmax(batched_img).cpu().detach().numpy())

    X = np.array(X)
    y = np.array(y)
    y_pred = np.array(y_pred)

    return X, y, y_pred

if __name__ == "__main__":
    # ---------------------
    # Setup for extracting training features
    # ---------------------

    output_folder_path = "/path/to/features/"
    training_folder_path = "/path/to/train/"
    model_class = GoogleNet
    label_filepath = get_all_classes_filepath()

    extract_features(
        output_folder_path=output_folder_path,
        training_folder_path=training_folder_path,
        model_class=model_class,
        label_filepath=label_filepath
    )

    # ---------------------
    # Setup for extracting validation features
    # ---------------------
    
    # output_folder_path = "/path/to/validation_features/"
    # validation_folder_path = "/path/to/Imagenet/val/"
    # model_class = AlexNet
    # label_filepath = get_animal_classes_filepath()

    # extract_validation_features(
    #     output_folder_path=output_folder_path,
    #     validation_folder_path=validation_folder_path,
    #     model_class=model_class,
    #     label_filepath=label_filepath
    # )