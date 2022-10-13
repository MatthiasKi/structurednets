import numpy as np
import torch
import copy

from structurednets.approximators.approximator import Approximator

class LDRApproximator(Approximator):
    def __init__(self, norm="fro"):
        self.possible_special_norms = ["nuc", "fro"]
        assert isinstance(norm, int) or isinstance(norm, float) or norm == np.inf or norm == -np.inf or (isinstance(norm, str) and norm in self.possible_special_norms), "Asked norm is not implemented: " + str(norm)

        self.norm = norm
        self.optimizer = torch.optim.SGD

    def get_name(self):
        name = "LDRApproximator_" + str(self.norm)
        return name

    def get_max_ld_rank_wrt_max_nb_free_parameters(self, max_nb_free_parameters: int, target_shape_mat: tuple):
        n = min(target_shape_mat)
        return int((max_nb_free_parameters - 2*(3*n - 2) + 4) / (2*n))

    def get_nb_free_parameters(self, displacement_rank: int, target_mat_shape: tuple):
        n = min(target_mat_shape)
        return displacement_rank * 2 * n + 2 * (3*n - 2) + 4

    def get_init_tridiagonal_matrix_with_corners_torch(self, shape: tuple):
        assert shape[0] == shape[1], "get_init_tridiagonal_matrix is only defined for square matrix shapes"
        tmp = np.expand_dims(np.arange(shape[0]), 1)
        upper_diagonal_indices = np.concatenate((tmp[1:]-1, tmp[1:]), axis=1)
        lower_diagonal_indices = np.concatenate((tmp[:-1]+1, tmp[:-1]), axis=1)
        diagonal_indices = np.concatenate((tmp, tmp), axis=1)
        corner_indices = np.array([[0, shape[0] - 1], [shape[0] - 1, 0]])
        tridiagonal_indices = np.concatenate((upper_diagonal_indices, lower_diagonal_indices, diagonal_indices, corner_indices), axis=0)
        glorot_std = np.sqrt(6.0 / (shape[0] + shape[1]))
        curr_mat = torch.sparse_coo_tensor([tridiagonal_indices[:,0], tridiagonal_indices[:,1]], np.random.uniform(-glorot_std, glorot_std, size=(3*shape[0],)), shape)
        curr_mat.requires_grad_()
        return curr_mat

    def build_weight_matrix_torch(self, representation_matrices: list, target_mat_shape: tuple):
        # NOTE: We assume that the first entry in the list is A, the second B, the third G and the last H
        res = torch.zeros(target_mat_shape)

        for i in range(representation_matrices[2].shape[1]):
            krylov_a_gi = torch.cat([torch.index_select(representation_matrices[2], 1, torch.LongTensor([i]))] + [torch.matmul(torch.matrix_power(representation_matrices[0].to_dense(), j), torch.index_select(representation_matrices[2], 1, torch.LongTensor([i]))) for j in range(1, target_mat_shape[0])], 1)
            krylov_bT_h_i = torch.cat([torch.index_select(representation_matrices[3], 1, torch.LongTensor([i]))] + [torch.matmul(torch.matrix_power(torch.transpose(representation_matrices[1].to_dense(), 0, 1), j), torch.index_select(representation_matrices[3], 1, torch.LongTensor([i]))) for j in range(1, target_mat_shape[0])], 1)

            res += torch.matmul(krylov_a_gi, krylov_bT_h_i.T)

        return res

    def init_representation_matrices_torch(self, weights: np.ndarray, displacement_rank: int):
        return [
            self.get_init_tridiagonal_matrix_with_corners_torch(weights.shape), # A
            self.get_init_tridiagonal_matrix_with_corners_torch(weights.shape), # B
            self.get_init_random_matrix_torch((weights.shape[0], displacement_rank)), # G
            self.get_init_random_matrix_torch((weights.shape[0], displacement_rank)), # H
        ]

    def approximate(self, optim_mat: np.ndarray, nb_params_share: float):
        assert len(optim_mat.shape) == 2, "Can only handle matrices for LDR approximation"

        best_ldr_mat = None
        best_approx_mat_dense = np.zeros_like(optim_mat)
        best_norm_difference = np.linalg.norm(optim_mat, ord=self.norm)
        nb_parameters = 0

        # This approach does only work for square matrices
        if optim_mat.shape[0] == optim_mat.shape[1]:
            max_nb_free_parameters = int(optim_mat.size * nb_params_share)
            displacement_rank = self.get_max_ld_rank_wrt_max_nb_free_parameters(max_nb_free_parameters=max_nb_free_parameters, target_shape_mat=optim_mat.shape)
            if displacement_rank > 0:
                nb_parameters = self.get_nb_free_parameters(displacement_rank, optim_mat.shape)

                # We repeat the optimization 5 times to account for random initialization effects
                for _ in range(5):
                    ldr_mat, approx_mat_dense = self.square_mat_ldr_approximation_torch(weights=optim_mat, displacement_rank=displacement_rank, verbose=False)
                
                    norm = np.linalg.norm(optim_mat - approx_mat_dense, self.norm)
                    if norm < best_norm_difference:
                        best_norm_difference = norm
                        best_ldr_mat = ldr_mat
                        best_approx_mat_dense = approx_mat_dense

        res_dict = dict()
        res_dict["type"] = "TDLDRApproximator"
        res_dict["ldr_mat"] = best_ldr_mat
        res_dict["approx_mat_dense"] = best_approx_mat_dense
        res_dict["nb_parameters"] = nb_parameters
        return res_dict

    def get_glorot_std(self, shape: tuple):
        if len(shape) == 2:
            glorot_std = np.sqrt(6.0 / (shape[0] + shape[1]))
        elif len(shape) == 1:
            glorot_std = np.sqrt(6.0 / shape[0])
        else:
            raise Exception("Shape length mismatch: " + str(len(shape)))
        return glorot_std

    def get_init_random_matrix_torch(self, shape: tuple):
        glorot_std = self.get_glorot_std(shape)
        curr_mat = torch.tensor(np.random.uniform(-glorot_std, glorot_std, size=shape))
        curr_mat.requires_grad_()
        return curr_mat

    def ldr_approx_loss_torch(self, representation_matrices: list, target_mat: np.ndarray):
        curr_mat = self.build_weight_matrix_torch(representation_matrices, target_mat.shape)
        return torch.norm(torch.tensor(target_mat) - curr_mat, p=self.norm)

    def square_mat_ldr_approximation_torch_with_lr(self, representation_matrices: list, weights: np.ndarray, lr: float, min_optimization_epsilon=1e-5, verbose=False):
        if verbose:
            print("Starting square mat ldr approximation with torch with learning rate " + str(lr))

        optimizer = self.optimizer(representation_matrices, lr=lr)
        last_loss = 1e9
        loss = 1e8
        optim_step = 1
        best_representation_matrices = copy.deepcopy(representation_matrices)
        while last_loss - loss > min_optimization_epsilon:
            last_loss = loss
            optimizer.zero_grad()
            loss = self.ldr_approx_loss_torch(representation_matrices=representation_matrices, target_mat=weights)
            if verbose:
                print('Optimization Step # {}, loss: {}'.format(optim_step, loss.item()))
            loss.backward()

            if loss < last_loss:
                best_representation_matrices = copy.deepcopy(representation_matrices)
        
            optimizer.step()
            optim_step += 1

        return best_representation_matrices

    def square_mat_ldr_approximation_torch(self, weights: np.ndarray, displacement_rank: int, min_optimization_epsilon=1e-3, verbose=False):
        assert weights.shape[0] == weights.shape[1], "square_mat_ldr_approximation is only defined for square matrices"

        representation_matrices = self.init_representation_matrices_torch(weights=weights, displacement_rank=displacement_rank)

        # Perform the optimization
        # NOTE there is the potential to run into numerical problems - then we return fresh initialized representation matrices (i.e. representation matrices which will perform pretty bad)
        try:
            # First we perform the optimization with a bigger learning rate
            representation_matrices = self.square_mat_ldr_approximation_torch_with_lr(representation_matrices=representation_matrices, weights=weights, lr=1, min_optimization_epsilon=min_optimization_epsilon, verbose=verbose)
            # And secondly we fine tune with a smaller learning rate
            representation_matrices = self.square_mat_ldr_approximation_torch_with_lr(representation_matrices=representation_matrices, weights=weights, lr=1e-1, min_optimization_epsilon=min_optimization_epsilon, verbose=verbose)
            # And again smaller
            representation_matrices = self.square_mat_ldr_approximation_torch_with_lr(representation_matrices=representation_matrices, weights=weights, lr=1e-2, min_optimization_epsilon=min_optimization_epsilon, verbose=verbose)
        except:
            representation_matrices = self.init_representation_matrices_torch(weights=weights, displacement_rank=displacement_rank)

        return representation_matrices, self.build_weight_matrix_torch(representation_matrices=representation_matrices, target_mat_shape=weights.shape).detach().numpy()

if __name__ == "__main__":
    # Test case
    optim_mat = np.random.uniform(-1, 1, size=(100, 100))
    test_norms = [-1, 1, -2, 2, np.inf, -np.inf, "nuc", "fro"]
    for norm in test_norms:
        approximator = LDRApproximator(norm=norm)
        res_dict = approximator.approximate(optim_mat=optim_mat, nb_params_share=0.8)
        norm_difference = np.linalg.norm(optim_mat - res_dict["approx_mat_dense"], ord="fro")
        print("--- Checked Norm: " + str(norm) + " ---")
        print("Norm Difference: " + str(norm_difference) + " (Original Mat Norm: " + str(np.linalg.norm(optim_mat, ord=norm)))

    halt = 1