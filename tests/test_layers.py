from unittest import TestCase
import numpy as np
import torch

from structurednets.layers.semiseparablelayer import SemiseparableLayer

class LayerTests(TestCase):
    def test_semiseparable_layer(self):
        nb_samples = 51
        nb_inputs = 4096
        nb_outputs = 100
        nb_params_share = 0.5
        nb_states = 50

        random_input = np.random.uniform(-1,1,size=(nb_samples, nb_inputs))

        layer = SemiseparableLayer(input_size=nb_inputs, output_size=nb_outputs, nb_params_share=nb_params_share, nb_states=nb_states)
        layer_output = layer(torch.tensor(random_input).float()).detach().numpy()

        T = layer.initial_T
        system_output = random_input @ T.T + layer.bias.detach().numpy()

        self.assertTrue(np.allclose(layer_output, system_output, atol=1e-5), "The layer output does not match the system output")