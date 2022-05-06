import torch
import torch.nn as nn
import numpy as np

from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

def get_nb_parameters(optim_mat: np.ndarray, statespace_dim: int, nb_states: int):
    dims_in, dims_out = standard_dims_in_dims_out_computation(input_size=optim_mat.shape[1], output_size=optim_mat.shape[0], nb_states=nb_states)
    nb_params = 0
    for state_i, (inp_dim, out_dim) in enumerate(zip(dims_in, dims_out)):
        if state_i > 0 and state_i < len(dims_in) - 1:
            nb_params += 2 * statespace_dim * statespace_dim
        nb_params += 2 * inp_dim * statespace_dim
        nb_params += 2 * statespace_dim * out_dim
        nb_params += inp_dim * out_dim
    return nb_params

def get_max_statespace_dim(optim_mat: np.ndarray, nb_params_share: float, nb_states: int):
    state_space_dim = 0
    while get_nb_parameters(optim_mat=optim_mat, statespace_dim=state_space_dim, nb_states=nb_states) < int(nb_params_share * optim_mat.size):
        state_space_dim += 1
    return state_space_dim
    
class SemiseparableLayer(nn.Module):
    def __init__(self, input_size, output_size,  nb_params_share: float, nb_states=1, initial_T=None, initial_bias=None):
        super(SemiseparableLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.nb_states = nb_states

        assert self.nb_states > 0, "Nb states must be positive"
        assert self.input_size > 0, "Input size must be positive"
        assert self.output_size > 0, "Output size must be positive"

        self.init_state_matrices(nb_params_share=nb_params_share, initial_T=initial_T, initial_bias=initial_bias)

    def init_state_matrices(self, nb_params_share: float, initial_T=None, initial_bias=None):
        if initial_T is None:
            T_full = get_random_glorot_uniform_matrix(shape=(self.output_size, self.input_size))
        else:
            T_full = initial_T

        self.statespace_dim = get_max_statespace_dim(T_full, nb_params_share=nb_params_share, nb_states=self.nb_states)

        self.dims_in, self.dims_out = standard_dims_in_dims_out_computation(input_size=self.input_size, output_size=self.output_size, nb_states=self.nb_states)
        T_operator = ToeplitzOperator(T_full, self.dims_in, self.dims_out)
        S = SystemIdentificationSVD(toeplitz=T_operator, max_states_local=self.statespace_dim)
        system_approx = MixedSystem(S)
        self.initial_T = system_approx.to_matrix()

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

        if initial_bias is None:
            self.bias = nn.parameter.Parameter(torch.tensor(get_random_glorot_uniform_matrix((self.output_size,))).float(), requires_grad=True)
        else:
            self.bias = nn.parameter.Parameter(torch.tensor(initial_bias), requires_grad=True)

    def get_input_index_range_according_to_state(self, state_i):
        return range(np.sum(self.dims_in[:state_i]), np.sum(self.dims_in[:state_i+1]))

    def get_output_index_range_according_to_state(self, state_i):
        return range(np.sum(self.dims_out[:state_i]), np.sum(self.dims_out[:state_i+1]))

    def forward(self, U):
        causal_state = torch.zeros((0, U.shape[0]))
        anticausal_state = torch.zeros((0, U.shape[0]))

        y_pred = torch.zeros((self.output_size, U.shape[0]))
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
        y_pred += self.bias
        return y_pred

def get_random_glorot_uniform_matrix(shape: tuple):
    limit = np.sqrt(6 / sum(shape))
    return np.random.uniform(-limit, limit, size=shape)

def standard_dims_in_dims_out_computation(input_size: int, output_size: int, nb_states: int):
    dims_in = int(input_size / nb_states) * np.ones((nb_states,), dtype='int32')
    dims_in[:(input_size - np.sum(dims_in))] += 1
    assert np.sum(dims_in) == input_size, "Sum over input dimensions does not match the input size"
    dims_out = int(output_size / nb_states) * np.ones((nb_states,), dtype='int32')
    dims_out[:(output_size - np.sum(dims_out))] += 1
    assert np.sum(dims_out) == output_size, "Sum over output dimensions does not match the output size"
    return dims_in, dims_out