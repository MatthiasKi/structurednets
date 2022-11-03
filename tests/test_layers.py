from unittest import TestCase
import numpy as np
import torch

from structurednets.layers.sss_layer import SSSLayer
from structurednets.layers.lr_layer import LRLayer
from structurednets.layers.psm_layer import PSMLayer
from structurednets.layers.ldr_layer import LDRLayer
from structurednets.layers.hmat_layer import HMatLayer
from structurednets.layers.layer_helpers import get_nb_model_parameters
from structurednets.training_helpers import train
from structurednets.approximators.psm_approximator import PSMApproximator
from structurednets.approximators.hmat_approximator import HMatApproximator
from structurednets.approximators.lr_approximator import LRApproximator
from structurednets.approximators.ldr_approximator import LDRApproximator
from structurednets.approximators.sss_approximator import SSSApproximator

def get_test_layer_classes(add_layers_requiring_square_weight_matrices=True) -> list:
    res = [
        LRLayer,
        PSMLayer,
        SSSLayer,
        HMatLayer,
    ]
    if add_layers_requiring_square_weight_matrices:
        res.append(LDRLayer)
    return res

class LayerTests(TestCase):
    def test_nb_parameters_are_correct(self):
        input_dim = 31
        output_dim = 20
        initial_weight_matrix = np.random.uniform(-1, 1, size=(output_dim, input_dim))

        layer_classes = get_test_layer_classes(add_layers_requiring_square_weight_matrices=False)
        nb_param_shares = np.linspace(0.1, 0.9, num=3)
        for nb_param_share in nb_param_shares:
            max_nb_parameters = int(nb_param_share * input_dim * output_dim)
            for layer_class in layer_classes:
                layer_from_scratch = layer_class(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share)
                nb_params = get_nb_model_parameters(layer_from_scratch)
                self.assertTrue(nb_params <= max_nb_parameters, "Layer " + str(layer_class) + " has too many parameters when building from scratch")

                layer_from_initial_weight_matrix = layer_class(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share, initial_weight_matrix=initial_weight_matrix)
                nb_params = get_nb_model_parameters(layer_from_initial_weight_matrix)
                self.assertTrue(nb_params <= max_nb_parameters, "Layer " + str(layer_class) + " has too many parameters when initializing from given weight matrix")

        # NOTE: we adapted the test for LDR matrices, because currently they are only implemented for square matrices
        input_dim = 20
        output_dim = 20
        initial_weight_matrix = np.random.uniform(-1, 1, size=(output_dim, input_dim))

        nb_param_shares = np.linspace(0.1, 0.9, num=3)
        for nb_param_share in nb_param_shares:
            layer = LDRLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share)
            nb_params = get_nb_model_parameters(layer)
            self.assertTrue(nb_params <= max_nb_parameters)

            layer = LDRLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share, initial_weight_matrix=initial_weight_matrix)
            nb_params = get_nb_model_parameters(layer)
            self.assertTrue(nb_params <= max_nb_parameters)

    def tes_nb_parameters_function_returns_correct_value(self):
        input_dim = 30
        output_dim = 30

        layer_classes = get_test_layer_classes()
        nb_param_shares = np.linspace(0.1, 0.9, num=3)
        for nb_param_share in nb_param_shares:
            for layer_class in layer_classes:
                layer = layer_class(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_param_share, use_bias=True)
                nb_params = get_nb_model_parameters(layer)
                nb_params_from_function = layer.get_nb_parameters()
                self.assertTrue(nb_params == nb_params_from_function, "The nuber of parameters do not match the value returned by the number of parameters function for the " + str(layer) + " layer")
            
    def test_train_improvement(self):
        input_dim = 20
        output_dim = 20
        nb_params_share = 0.5

        nb_training_samples = 1000
        train_input = np.random.uniform(-1, 1, size=(nb_training_samples, input_dim)).astype(np.float32)
        train_input_torch = torch.tensor(train_input).float()
        train_output = np.ones((nb_training_samples, output_dim), dtype=np.float32)

        layer_classes = get_test_layer_classes()
        for layer_class in layer_classes:
            layer = layer_class(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share)

            pred = layer.forward(train_input_torch).detach().numpy()
            max_error_before = np.max(np.abs(train_output - pred))

            # NOTE that the arguments are specifically chosen such that the training only lasts for a few epochs
            trained_layer, _, _, _, _, _, _, _, _ = train(
                model=layer, X_train=train_input, y_train=train_output,
                patience=10, batch_size=nb_training_samples, verbose=False, lr=1e-3, 
                restore_best_model=False, loss_function_class=torch.nn.MSELoss,
                min_patience_improvement=1e6, optimizer_class=torch.optim.SGD,
            )

            pred = trained_layer.forward(train_input_torch).detach().numpy()
            max_error_after = np.max(np.abs(train_output - pred))

            self.assertTrue(max_error_after < max_error_before, str(layer) + " failed to improve the error")

    def test_sss_layer(self):
        nb_samples = 51
        nb_inputs = 76
        nb_outputs = 14
        nb_params_share = 0.5
        nb_states = 10

        random_input = np.random.uniform(-1,1,size=(nb_samples, nb_inputs))

        layer = SSSLayer(input_dim=nb_inputs, output_dim=nb_outputs, nb_params_share=nb_params_share, nb_states=nb_states)
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

    def test_psm_layer(self):
        input_dim = 50
        output_dim = 50

        random_mat = np.random.uniform(-1, 1, size=(50, 50)).astype(np.float32)
        approximator = PSMApproximator(nb_matrices=2, linear_nb_nonzero_elements_distribution=True)
        res_dict = approximator.approximate(optim_mat=random_mat, nb_params_share=0.2, num_interpolation_steps=3)

        random_input = np.random.uniform(-1, 1, size=(100, 50)).astype(np.float32)
        target_output = (res_dict["approx_mat_dense"] @ random_input.T).T

        layer = PSMLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, sparse_matrices=res_dict["faust_approximation"])
        pred_output = layer.forward(torch.tensor(random_input)).detach().numpy()

        self.assertTrue(np.allclose(pred_output, target_output, atol=1e-5, rtol=1e-5))

    def test_recovering_from_res_dicts(self):
        input_dim = 50
        output_dim = 50
        nb_params_share = 0.5

        optim_mat = np.random.uniform(-1, 1, size=(output_dim, input_dim)).astype(np.float32)
        random_input = np.random.uniform(-1, 1, size=(10, input_dim)).astype(np.float32)
        random_input_torch = torch.tensor(random_input)

        approximator = LRApproximator()
        res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share)
        approximator_pred = (res_dict["approx_mat_dense"] @ random_input.T).T
        layer = LRLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share, use_bias=False, initial_lr_components=[res_dict["left_mat"], res_dict["right_mat"]])
        layer_pred = layer.forward(random_input_torch).detach().numpy()
        self.assertTrue(np.allclose(approximator_pred, layer_pred, atol=1e-5, rtol=1e-5))

        approximator = LDRApproximator()
        res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share)
        approximator_pred = (res_dict["approx_mat_dense"] @ random_input.T).T
        layer = LDRLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share, use_bias=False, initial_representation_matrices=res_dict["ldr_mat"])
        layer_pred = layer.forward(random_input_torch).detach().numpy()
        self.assertTrue(np.allclose(approximator_pred, layer_pred, atol=1e-5, rtol=1e-5))

        approximator = HMatApproximator()
        res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share)
        approximator_pred = (res_dict["approx_mat_dense"] @ random_input.T).T
        layer = HMatLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share, use_bias=False, initial_hmatrix=res_dict["h_matrix"])
        layer_pred = layer.forward(random_input_torch).detach().numpy()
        self.assertTrue(np.allclose(approximator_pred, layer_pred, atol=1e-5, rtol=1e-5))

        approximator = PSMApproximator(nb_matrices=2, linear_nb_nonzero_elements_distribution=True)
        res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share)
        approximator_pred = (res_dict["approx_mat_dense"] @ random_input.T).T
        layer = PSMLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share, use_bias=False, sparse_matrices=res_dict["faust_approximation"])
        layer_pred = layer.forward(random_input_torch).detach().numpy()
        self.assertTrue(np.allclose(approximator_pred, layer_pred, atol=1e-5, rtol=1e-5))

        nb_states = 5
        approximator = SSSApproximator(nb_states=nb_states)
        res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=nb_params_share)
        approximator_pred = (res_dict["approx_mat_dense"] @ random_input.T).T
        layer = SSSLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share, use_bias=False, nb_states=nb_states, initial_system_approx=res_dict["system_approx"])
        layer_pred = layer.forward(random_input_torch).detach().numpy()
        self.assertTrue(np.allclose(approximator_pred, layer_pred, atol=1e-5, rtol=1e-5))