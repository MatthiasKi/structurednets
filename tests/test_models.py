from unittest import TestCase
import torch
import numpy as np

from structurednets.models.alexnet import AlexNet
from structurednets.models.googlenet import GoogleNet
from structurednets.models.inceptionv3 import InceptionV3
from structurednets.models.mobilenetv2 import MobilenetV2
from structurednets.models.resnet18 import Resnet18
from structurednets.models.vgg16 import VGG16

class ModelTests(TestCase):
    def test_model_functions(self):
        model_classes = [AlexNet, GoogleNet, InceptionV3, MobilenetV2, Resnet18, VGG16]
        
        for model_class in model_classes:
            print("Checking " + model_class.__name__)

            net = model_class([1,2,244,654], use_gpu=False)

            sample_batch = torch.tensor(np.random.uniform(-1,1,size=(10, 3, 255, 255))).float()

            features = net.get_features_for_batch(sample_batch)
            optim_mat = net.get_optimization_matrix()

            net.predict_pretrained_with_argmax(sample_batch)

            pretrained_output = net.predict_pretrained(sample_batch)
            featured_output = net.features_and_optim_mat_to_prediction(features, optim_mat)
            self.assertTrue(torch.allclose(pretrained_output, featured_output, rtol=1e-4, atol=1e-7), "Mismatch between featured and standard output")