# "Feature" Extraction

The application example used in our scripts is image recognition based on the Imagenet dataset. Since this dataset is very large (it contains more than a million images, with 1000 classes in total), it is often not feasible to train a whole model again and again to compare the effect of using different matrix structures in the model. However, we only focus on the last, densely connect layer of the deep pretrained models (which most often contains most of the parameters of the overall network, since parameters in the convolutional parts are shared). Therefore, we can freeze the first layers of the model and don't need to compute them over and over again during the training. 

In fact, we only need to compute the activations of the network once, up to the layer which we want to modify. We call the inputs to the layer we want to modify "features". In order to compute these features, the last (densely connected) layer is removed from the network and the activations are stored into a file. The extraction can be done using the extract_features() function in structurednets.features.extract_features. The feature extraction can afterwards be checked using the check_features() function in structurednets.features.check_feature_extraction. 

## Usage

First, you need to download the [Imagenet Images](https://www.image-net.org/). In our examples, we used the 2012 version of the Imagenet data. Then you can extract the features (training and validation features seperately) using the following functions:
- `extract_features(output_folder_path: str, training_folder_path: str, model_class: VisionModel, label_filepath: str, use_gpu=True)`
- `extract_validation_features(output_folder_path: str, validation_folder_path: str, model_class: VisionModel, label_filepath: str, use_gpu=True)`

Here, the `output_folder_path` is the path to the folder in which the feature file will be placed (using `pickle`). The `training_folder_path` / `validation_folder_path` is the path to the folder where the downloaded Imagenet images lie (i.e. the `train/` or `/val` folder of the extracted archive). The model class is one of the classes defined in the [model class folder](https://github.com/MatthiasKi/structurednets/tree/master/src/structurednets/models) (i.e. an implementation of the abstract `VisionModel` class). This model is used to generate the features (the last layer is removed, and the outputs of the second last layer are stored as "features" when propagating the images in the respective folder through the model). The label filepath is the filepath of the label file, which can be for example obtained using the `asset_helpers.py` of this package:
- `asset_helpers.get_all_classes_filepath`: Returns the label filepath for all 1000 Imagenet classes
- `asset_helpers.get_animal_classes_filepath`: Returns the label filepath for 100 animal classes selected from the Imagenet classes
- `asset_helpers.get_object_classes_filepath`: Returns the label filepath for 100 object classes selected from the Imagenet classes

Use the `use_gpu` argument to indicate if the GPU should be used when computing the features. 

## Features Format

The features are stored as pickled tuple. Note that you have to read the pickle file in binary format, i.e. `pickle.load(open("path/to/features.p", "rb"))`. The tuple contains 3 entries: `[X, y, y_pred]`:
- `X` are the extracted features of the shape `(number of images, 1, number of neurons in the second last layer of the model used for extraction)`
- `y` which are the true labels for each of the features. `y` has the shape `(number of images,)`
- `y_pred` which are labels predicted by the model used for extraction for each of the features. This has the shape `(number of images, 1)`

## Feature checking

We provided some scripts to analyze the Imagenet data and to check if the feature extraction was successfull:
- `features/imagenet_stats/print_imagenet_data_stats(label_filepath: str, training_folder_path: str)`: This function can be used to get some stats of the images in the training folder (for the chosen label file)
- `features/imagenet_stats/check_feature_extraction/check_features(feature_filepath: str, model_class: VisionModel, label_filepath: str)`: This function can be used to check the feature extraction. It checks if the shapes of the extracted arrays match, and prints the accuracy for each class (please make manually sure that there is no outlier here). 