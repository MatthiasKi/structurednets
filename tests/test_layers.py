from unittest import TestCase
import numpy as np
import torch

from structurednets.layers.sss_layer import SemiseparableLayer
from structurednets.layers.lr_layer import LRLayer
from structurednets.layers.psm_layer import PSMLayer
from structurednets.layers.layer_helpers import get_nb_model_parameters

class LayerTests(TestCase):
    def test_nb_parameters_are_correct(self):
        input_dim = 31
        output_dim = 20
        initial_weight_matrix = np.random.uniform(-1, 1, size=(output_dim, input_dim))

        nb_param_shares = np.linspace(0.1, 0.9, num=3)
        for nb_param_share in nb_param_shares:
            max_nb_parameters = int(nb_param_share * input_dim * output_dim)

            layer = LRLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share)
            nb_params = get_nb_model_parameters(layer)
            self.assertTrue(nb_params <= max_nb_parameters)

            layer = LRLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share, initial_weight_matrix=initial_weight_matrix)
            nb_params = get_nb_model_parameters(layer)
            self.assertTrue(nb_params <= max_nb_parameters)

            layer = PSMLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share)
            nb_params = get_nb_model_parameters(layer)
            self.assertTrue(nb_params <= max_nb_parameters)

            layer = PSMLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share, initial_weight_matrix=initial_weight_matrix)
            nb_params = get_nb_model_parameters(layer)
            self.assertTrue(nb_params <= max_nb_parameters)

            layer = SemiseparableLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share, nb_states=5)
            nb_params = get_nb_model_parameters(layer)
            self.assertTrue(nb_params <= max_nb_parameters)

            layer = SemiseparableLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share, initial_weight_matrix=initial_weight_matrix, nb_states=5)
            nb_params = get_nb_model_parameters(layer)
            self.assertTrue(nb_params <= max_nb_parameters)

    def test_train_to_zero(self):
        # TODO add a test for checking that all layers can be trained to learn to output only zeros
        pass

    def test_semiseparable_layer(self):
        nb_samples = 51
        nb_inputs = 76
        nb_outputs = 14
        nb_params_share = 0.5
        nb_states = 10

        random_input = np.random.uniform(-1,1,size=(nb_samples, nb_inputs))

        layer = SemiseparableLayer(input_dim=nb_inputs, output_dim=nb_outputs, nb_params_share=nb_params_share, nb_states=nb_states)
        layer_output = layer(torch.tensor(random_input).float()).detach().numpy()

        T = layer.initial_weight_matrix
        system_output = random_input @ T.T + layer.bias.detach().numpy()

        self.assertTrue(np.allclose(layer_output, system_output, atol=1e-5), "The layer output does not match the system output")

    def test_lr_layer(self):
        nb_samples = 51
        nb_inputs = 4096
        nb_outputs = 100
        nb_params_share = 0.5

        max_nb_parameters = int(nb_params_share * nb_inputs * nb_outputs)
        rank = int(max_nb_parameters / (nb_inputs + nb_outputs))
        random_input = np.random.uniform(-1, 1, size=(nb_samples, nb_inputs))
        random_input_torch = torch.tensor(random_input).float()
        weight_matrix = np.random.uniform(-1, 1, size=(nb_outputs, rank)) @ np.random.uniform(-1, 1, (rank, nb_inputs))
        bias = np.random.uniform(-1, 1, size=(nb_outputs,))

        layer = LRLayer(input_dim=nb_inputs, output_dim=nb_outputs, nb_params_share=nb_params_share, use_bias=True, initial_weight_matrix=weight_matrix, initial_bias=bias)
        pred = layer.forward(random_input_torch).detach().numpy()

        target = (weight_matrix @ random_input.T).T + bias
        self.assertTrue(np.allclose(pred, target, rtol=1e-3, atol=1e-3), "The output of the LR layer differs from the expected output")

        sample_to_investigate = 12
        sample_input = random_input[sample_to_investigate, :]
        sample_target = weight_matrix @ sample_input + bias
        sample_pred = pred[sample_to_investigate]
        self.assertTrue(np.allclose(sample_target, sample_pred, rtol=1e-3, atol=1e-3), "The picked sample prediction should match the target")