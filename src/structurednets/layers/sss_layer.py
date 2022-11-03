import torch
import torch.nn as nn
import numpy as np
import pickle

from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

from structurednets.layers.layer_helpers import get_random_glorot_uniform_matrix, get_nb_model_parameters
from structurednets.training_helpers import train_with_decreasing_lr
from structurednets.layers.structured_layer import StructuredLayer

def get_nb_parameters(optim_mat_shape: tuple, statespace_dim: int, nb_states: int) -> int:
    dims_in, dims_out = standard_dims_in_dims_out_computation(input_size=optim_mat_shape[1], output_size=optim_mat_shape[0], nb_states=nb_states)
    nb_params = 0
    for state_i, (inp_dim, out_dim) in enumerate(zip(dims_in, dims_out)):
        if state_i > 0 and state_i < len(dims_in) - 1:
            nb_params += 2 * statespace_dim * statespace_dim
        nb_params += 2 * inp_dim * statespace_dim
        nb_params += 2 * statespace_dim * out_dim
        nb_params += inp_dim * out_dim
    return nb_params

def get_max_statespace_dim(optim_mat: np.ndarray, nb_params_share: float, nb_states: int) -> int:
    state_space_dim = 0
    while has_less_parameters_than_allowed(optim_mat=optim_mat, nb_params_share=nb_params_share, nb_states=nb_states, state_space_dim=state_space_dim+1):
        state_space_dim += 1
    return state_space_dim

def has_less_parameters_than_allowed(optim_mat: np.ndarray, nb_params_share: float, nb_states: int, state_space_dim: int) -> bool:
    return get_nb_parameters(optim_mat_shape=optim_mat.shape, statespace_dim=state_space_dim, nb_states=nb_states) < int(nb_params_share * optim_mat.size)
    
class SSSLayer(StructuredLayer):
    def __init__(self, input_dim: int, output_dim: int, nb_params_share: float, use_bias=True, initial_weight_matrix=None, initial_bias=None, nb_states=None, initial_system_approx=None):
        super(SSSLayer, self).__init__(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share, use_bias=use_bias, initial_weight_matrix=initial_weight_matrix, initial_bias=initial_bias)

        if nb_states is None:
            nb_states = int(min(input_dim, output_dim) / 2)
        assert nb_states > 0, "Nb states must be positive"

        assert initial_weight_matrix is None or initial_system_approx is None, "Either pass an initial weight matrix or the initial system approximation - not both"
        if initial_system_approx is not None:
            assert len(initial_system_approx.dims_in) == nb_states, "The given system approximation does not match the expected number of states (dims_in)"
            assert len(initial_system_approx.dims_out) == nb_states, "The given system approximation does not match the expected number of states (dims_out)"

        self.input_dim = input_dim
        self.nb_states = nb_states

        self.init_state_matrices(nb_params_share=nb_params_share, initial_weight_matrix=initial_weight_matrix, initial_system_approx=initial_system_approx)

    def init_state_matrices(self, nb_params_share: float, initial_weight_matrix=None, initial_system_approx=None):
        if initial_weight_matrix is None:
            T_full = get_random_glorot_uniform_matrix(shape=(self.output_dim, self.input_dim))
        else:
            T_full = initial_weight_matrix

        self.statespace_dim = get_max_statespace_dim(T_full, nb_params_share=nb_params_share, nb_states=self.nb_states)

        if has_less_parameters_than_allowed(optim_mat=T_full, nb_params_share=nb_params_share, nb_states=self.nb_states, state_space_dim=self.statespace_dim):
            self.dims_in, self.dims_out = standard_dims_in_dims_out_computation(input_size=self.input_dim, output_size=self.output_dim, nb_states=self.nb_states)
            
            if initial_system_approx is None:
                T_operator = ToeplitzOperator(T_full, self.dims_in, self.dims_out)
                S = SystemIdentificationSVD(toeplitz=T_operator, max_states_local=self.statespace_dim)
                system_approx = MixedSystem(S)
            else:
                system_approx = pickle.loads(pickle.dumps(initial_system_approx))

            self.initial_weight_matrix = system_approx.to_matrix()

            A = [stage.A_matrix for stage in system_approx.causal_system.stages]
            B = [stage.B_matrix for stage in system_approx.causal_system.stages]
            C = [stage.C_matrix for stage in system_approx.causal_system.stages]
            D = [stage.D_matrix for stage in system_approx.causal_system.stages]
            E = [stage.A_matrix for stage in system_approx.anticausal_system.stages]
            F = [stage.B_matrix for stage in system_approx.anticausal_system.stages]
            G = [stage.C_matrix for stage in system_approx.anticausal_system.stages]
            

            self.A = nn.ParameterList([nn.Parameter(torch.tensor(A_k).float(), requires_grad=True) for A_k in A])
            self.B = nn.ParameterList([nn.Parameter(torch.tensor(B_k).float(), requires_grad=True) for B_k in B])
            self.C = nn.ParameterList([nn.Parameter(torch.tensor(C_k).float(), requires_grad=True) for C_k in C])
            self.D = nn.ParameterList([nn.Parameter(torch.tensor(D_k).float(), requires_grad=True) for D_k in D])
            self.E = nn.ParameterList([nn.Parameter(torch.tensor(E_k).float(), requires_grad=True) for E_k in E])
            self.F = nn.ParameterList([nn.Parameter(torch.tensor(F_k).float(), requires_grad=True) for F_k in F])
            self.G = nn.ParameterList([nn.Parameter(torch.tensor(G_k).float(), requires_grad=True) for G_k in G])

            self.state_matrices_initialized = True

    def get_input_index_range_according_to_state(self, state_i):
        return range(np.sum(self.dims_in[:state_i]), np.sum(self.dims_in[:state_i+1]))

    def get_output_index_range_according_to_state(self, state_i):
        return range(np.sum(self.dims_out[:state_i]), np.sum(self.dims_out[:state_i+1]))

    def forward(self, U):
        if hasattr(self, "state_matrices_initialized") and self.state_matrices_initialized:
            y_pred = torch.zeros((self.output_dim, U.shape[0]))

            causal_state = torch.zeros((0, U.shape[0]))
            anticausal_state = torch.zeros((0, U.shape[0]))

            for causal_state_i in range(self.nb_states):
                causal_state_input_index_range = self.get_input_index_range_according_to_state(causal_state_i)
                causal_state_output_index_range = self.get_output_index_range_according_to_state(causal_state_i)
                u_causal = torch.transpose(U, 1, 0)[causal_state_input_index_range, :]
                y_pred[causal_state_output_index_range, :] += torch.matmul(self.C[causal_state_i], causal_state) + torch.matmul(self.D[causal_state_i], u_causal)
                causal_state = torch.matmul(self.A[causal_state_i], causal_state) + torch.matmul(self.B[causal_state_i], u_causal)

                anticausal_state_i = self.nb_states-1-causal_state_i
                anti_causal_input_index_range = self.get_input_index_range_according_to_state(anticausal_state_i)
                anti_causal_output_index_range = self.get_output_index_range_according_to_state(anticausal_state_i)
                u_anticausal = torch.transpose(U, 1, 0)[anti_causal_input_index_range, :]
                y_pred[anti_causal_output_index_range, :] += torch.matmul(self.G[anticausal_state_i], anticausal_state)
                anticausal_state = torch.matmul(self.E[anticausal_state_i], anticausal_state) + torch.matmul(self.F[anticausal_state_i], u_anticausal)

            y_pred = torch.transpose(y_pred, 1, 0)
            if self.use_bias:
                y_pred += self.bias

            return y_pred
        else:
            return torch.zeros((U.shape[0], self.output_dim))

    def get_nb_parameters(self) -> int:
        if hasattr(self, "state_matrices_initialized") and self.state_matrices_initialized:
            return get_nb_parameters(optim_mat_shape=(self.output_dim, self.input_dim), statespace_dim=self.statespace_dim, nb_states=self.nb_states)
        else:
            return 0

def standard_dims_in_dims_out_computation(input_size: int, output_size: int, nb_states: int):
    dims_in = int(input_size / nb_states) * np.ones((nb_states,), dtype='int32')
    dims_in[:(input_size - np.sum(dims_in))] += 1
    assert np.sum(dims_in) == input_size, "Sum over input dimensions does not match the input size"
    dims_out = int(output_size / nb_states) * np.ones((nb_states,), dtype='int32')
    dims_out[:(output_size - np.sum(dims_out))] += 1
    assert np.sum(dims_out) == output_size, "Sum over output dimensions does not match the output size"
    return dims_in, dims_out

if __name__ == "__main__":
    input_dim = 31
    output_dim = 20
    nb_params_share = 0.5

    nb_training_samples = 1000
    train_input = np.random.uniform(-1, 1, size=(nb_training_samples, input_dim)).astype(np.float32)
    train_output = np.ones((nb_training_samples, output_dim), dtype=np.float32)

    layer = SSSLayer(input_dim=input_dim, output_dim=output_dim, nb_params_share=nb_params_share)
    trained_layer, start_train_loss, start_train_accuracy, start_val_loss, start_val_accuracy, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train_with_decreasing_lr(
        model=layer, X_train=train_input, y_train=train_output,
        patience=1, batch_size=nb_training_samples, verbose=False,
        loss_function_class=torch.nn.MSELoss,
        min_patience_improvement=1e5 # This makes the model being trained for only few epochs
    )

    train_input_torch = torch.tensor(train_input).float()
    pred = trained_layer.forward(train_input_torch).detach().numpy()

    max_error = np.max(np.abs(pred - train_output))

    # ---

    input_dim = 51
    output_dim = 40
    initial_weight_matrix = np.random.uniform(-1, 1, size=(output_dim, input_dim))

    nb_param_share = 0.5
    max_nb_parameters = int(nb_param_share * input_dim * output_dim)
    min_nb_parameters = int((nb_param_share - 0.1) * input_dim * output_dim)
    
    layer = SSSLayer(input_dim=input_dim, output_dim=output_dim, use_bias=False, nb_params_share=nb_param_share, nb_states=5)
    nb_params = get_nb_model_parameters(layer)

    assert nb_params <= max_nb_parameters, "too many parameters"

    U = torch.tensor(np.random.uniform(-1, 1, size=(50, input_dim))).float()
    res = layer.forward(U)

    halt = 1