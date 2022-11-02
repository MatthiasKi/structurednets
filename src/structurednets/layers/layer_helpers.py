import torch
import torch.nn as nn
import numpy as np

def get_random_glorot_uniform_matrix_torch(shape: tuple) -> torch.tensor:
    curr_mat = torch.tensor(get_random_glorot_uniform_matrix(shape=shape))
    curr_mat.requires_grad_()
    return curr_mat
    
def get_random_glorot_uniform_matrix(shape: tuple) -> np.ndarray:
    limit = np.sqrt(6 / sum(shape))
    return np.random.uniform(-limit, limit, size=shape)

def get_nb_model_parameters(model: nn.Module, count_gradientless_parameters=True):
    nb_parameters = 0
    for name, param in model.named_parameters():
        if param.requires_grad or count_gradientless_parameters:
            if param.is_sparse:
                nb_parameters += param._nnz()
            else:
                nb_parameters += param.numel()
    return nb_parameters