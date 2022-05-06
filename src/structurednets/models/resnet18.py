from torchvision.models import resnet18
import torch
import numpy as np
from copy import deepcopy

from structurednets.models.visionmodel import VisionModel, get_nb_parameters_in_model

class Resnet18(VisionModel):
    def __init__(self, output_indices: list, use_gpu=True):
        self.model = resnet18(pretrained=True).eval()

        self.feature_model = deepcopy(self.model)
        self.feature_model.fc = torch.nn.Identity()
        self.feature_model.eval()

        super().__init__(output_indices, use_gpu=use_gpu)

    def get_optimization_matrix(self):
        return self.model.fc.weight[self.output_indices,:].T

    def get_bias(self):
        return self.model.fc.bias[self.output_indices]

    def get_feature_extractor_nb_params(self):
        return get_nb_parameters_in_model(self.feature_model)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Resnet18([1,2,100,654])

    nb_feature_extractor_params = net.get_feature_extractor_nb_params()

    sample_batch = torch.tensor(np.random.uniform(-1,1,size=(10, 3, 255, 255))).float()
    sample_batch = sample_batch.to(device)

    features = net.get_features_for_batch(sample_batch)
    optim_mat = net.get_optimization_matrix()

    net.predict_pretrained_with_argmax(sample_batch)

    pretrained_output = net.predict_pretrained(sample_batch)
    featured_output = net.features_and_optim_mat_to_prediction(features, optim_mat)
    assert torch.allclose(pretrained_output, featured_output), "Mismatch between featured and standard output"

    halt = 1