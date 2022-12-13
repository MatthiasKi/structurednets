import torch
import numpy as np
import abc

def get_device(use_gpu=True):
    if use_gpu:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return 'cpu'

def get_nb_parameters_in_model(model: torch.nn, count_gradientless_parameters=True) -> int:
    nb_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad or count_gradientless_parameters:
            nb_parameters += param.numel()
    return nb_parameters

class VisionModel:
    __metaclass__ = abc.ABCMeta

    def __init__(self, output_indices: list, use_gpu=True):
        self.device = get_device(use_gpu=use_gpu)   
        self.model.to(self.device)
        self.feature_model.to(self.device)
        self.output_indices = np.array(output_indices)

    def get_features_for_batch(self, X: torch.tensor):
        return self.feature_model(X)

    def predict_pretrained_with_argmax(self, X: torch.tensor):
        return self.predict_pretrained(X).argmax(axis=1)

    def predict_pretrained(self, X: torch.tensor):
        return self.model(X)[:, self.output_indices]

    def features_and_optim_mat_to_prediction(self, features: torch.tensor, optim_mat: torch.tensor):
        return torch.matmul(features, optim_mat) + self.get_bias()

    def features_and_optim_mat_to_prediction_with_argmax(self, features: torch.tensor, optim_mat: torch.tensor):
        return self.features_and_optim_mat_to_prediction(features=features, optim_mat=optim_mat).argmax(axis=1)

    @abc.abstractmethod
    def get_optimization_matrix(self):
        return

    @abc.abstractmethod
    def get_bias(self):
        return

    @abc.abstractmethod
    def get_feature_extractor_nb_params(self):
        return