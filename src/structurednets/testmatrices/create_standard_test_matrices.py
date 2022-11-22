import numpy as np
import pickle
from sklearn.utils.extmath import randomized_svd

NB_MATRICES = 3
MATRIX_SIZE = 300

def mat_to_standard_format(mat: np.ndarray) -> np.ndarray:
    return mat.astype(np.float32)

def create_orthogonal_matrices():
    res = dict()
    res["name"] = "orthogonal_matrices"

    for matrix_idx in range(NB_MATRICES):
        rand_mat = np.random.uniform(-1, 1, size=(MATRIX_SIZE, MATRIX_SIZE))

        U, _, Vh = randomized_svd(M=rand_mat, n_components=MATRIX_SIZE)
        mat = U @ Vh

        res[matrix_idx] = dict()
        res[matrix_idx]["mat"] = mat_to_standard_format(mat)

        assert np.allclose(mat.T @ mat, np.eye(MATRIX_SIZE)), "produced test matrix which is not orthogonal"

    pickle.dump(res, open("orthogonal_matrices.p", "wb"))

def create_random_matrices():
    res = dict()
    res["name"] = "random_matrices"

    for matrix_idx in range(NB_MATRICES):
        mat = np.random.uniform(-1, 1, size=(MATRIX_SIZE, MATRIX_SIZE))
        res[matrix_idx] = dict()
        res[matrix_idx]["mat"] = mat_to_standard_format(mat)

    pickle.dump(res, open("random_matrices.p", "wb"))

def create_low_rank_matrices(rank=100):
    res = dict()
    res["name"] = "low_rank_matrices"

    for matrix_idx in range(NB_MATRICES):
        left = np.random.uniform(-1, 1, size=(MATRIX_SIZE, rank))
        right = np.random.uniform(-1, 1, size=(rank, MATRIX_SIZE))
        mat = left @ right

        res[matrix_idx] = dict()
        res[matrix_idx]["mat"] = mat_to_standard_format(mat)

        assert np.linalg.matrix_rank(mat) == rank, "produced test matrix with wrong rank"

    pickle.dump(res, open("low_rank_matrices.p", "wb"))

def create_distributed_singular_values_matrices():
    res = dict()
    res["name"] = "distributed_singular_values_matrices"

    for matrix_idx in range(NB_MATRICES):
        rand_mat = np.random.uniform(-1, 1, size=(MATRIX_SIZE, MATRIX_SIZE))

        U, _, Vh = randomized_svd(M=rand_mat, n_components=MATRIX_SIZE)
        s = np.linspace(0.1, 1.0, num=MATRIX_SIZE)[::-1]

        left_side = U @ np.diag(np.sqrt(s))
        right_side = np.diag(np.sqrt(s)) @ Vh
        mat = left_side @ right_side

        res[matrix_idx] = dict()
        res[matrix_idx]["mat"] = mat_to_standard_format(mat)

    pickle.dump(res, open("distributed_singular_values_matrices.p", "wb"))

if __name__ == "__main__":
    create_low_rank_matrices()
    create_orthogonal_matrices()
    create_random_matrices()
    create_distributed_singular_values_matrices()