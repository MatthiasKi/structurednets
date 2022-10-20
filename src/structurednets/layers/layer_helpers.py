import numpy as np

def get_random_glorot_uniform_matrix(shape: tuple):
    limit = np.sqrt(6 / sum(shape))
    return np.random.uniform(-limit, limit, size=shape)