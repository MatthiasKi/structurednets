import numpy as np
import pickle
from scipy.sparse import rand
from scipy.linalg import toeplitz

from structurednets.testmatrices.create_standard_test_matrices import mat_to_standard_format
from structurednets.approximators.hmat_approximator import build_hmat_block_cluster_tree
from structurednets.hmatrix.hmatrix import HMatrix

from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

NB_MATRICES = 3 # DO NOT CHANGE THIS (it is hardcoded in some functions - for example for creating the hierarchical matrices)
MATRIX_SIZE = 300

def create_random_sss_matrix(state_space_dim: int):
    orig = np.random.uniform(-1, 1, size=(MATRIX_SIZE, MATRIX_SIZE))

    inp_out_dim = 10
    nb_dims = int(MATRIX_SIZE / inp_out_dim)
    assert nb_dims == MATRIX_SIZE / inp_out_dim, "The MATRIX_SIZE must be a multiple of the inp_out_dim"
    dims_in = inp_out_dim * np.ones(shape=(nb_dims,), dtype='int32')
    dims_out = inp_out_dim * np.ones(shape=(nb_dims,), dtype='int32')

    T_operator = ToeplitzOperator(orig, dims_in, dims_out)
    S = SystemIdentificationSVD(toeplitz=T_operator, max_states_local=state_space_dim)
    system_approx = MixedSystem(S)
    res = system_approx.to_matrix()
    assert res.shape[0] == MATRIX_SIZE, "The produced matrix has the wrong shape"

    return res.astype(np.float32)

def create_sss_test_matrices():
    res = dict()
    res["name"] = "sss_matrices"

    for idx in range(NB_MATRICES):
        res[idx] = dict()
        mat = create_random_sss_matrix(state_space_dim=5)
        res[idx]["mat"] = mat_to_standard_format(mat)
    
    pickle.dump(res, open("sss_matrices.p", "wb"))

def get_random_sparse_matrix():
    return rand(MATRIX_SIZE, MATRIX_SIZE, density=0.1, format='csr')

def create_psm_test_matrices():
    res = dict()
    res["name"] = "psm_matrices"

    for matrix_idx in range(NB_MATRICES):
        mat1 = get_random_sparse_matrix()
        mat2 = get_random_sparse_matrix()
        mat3 = get_random_sparse_matrix()
        mat = (mat1 @ mat2 @ mat3).todense()

        res[matrix_idx] = dict()
        res[matrix_idx]["mat"] = mat_to_standard_format(mat)

        assert np.sum(np.abs(mat) < 1e-6) < mat.size * 0.1, "created a psm matrix with too many zero elements"
    
    pickle.dump(res, open("psm_matrices.p", "wb"))

def assert_square(mat: np.ndarray):
    assert mat.shape[0] == mat.shape[1], "The produced matrix should be square"

def assert_is_matrix(mat: np.ndarray):
    assert len(mat.shape) == 2, "The outcome should be a matrix"

def assert_is_rank_1(mat: np.ndarray):
    assert np.linalg.matrix_rank(mat) == 1, "The matrix is expected to have rank 1"

def get_rank_1_mat(matrix_size: int):
    res = np.random.uniform(-1, 1, size=(matrix_size, 1)) @ np.random.uniform(-1, 1, size=(1, matrix_size))
    assert_is_matrix(mat=res)
    assert_square(mat=res)
    assert res.shape[0] == matrix_size, "Mismatch between the desired matrix size and the size of the produced matrix"
    return res

def get_random_1d_hmat():
    random_mat = np.random.uniform(-1, 1, size=(MATRIX_SIZE, MATRIX_SIZE))
    block_cluster_tree = build_hmat_block_cluster_tree(matrix_shape=(MATRIX_SIZE, MATRIX_SIZE), eta=0.5)
    hmat = HMatrix(block_cluster_tree=block_cluster_tree, shape=random_mat.shape)
    hmat.set_full_components(random_mat)
    leaf_elements = hmat.block_cluster_tree.get_all_leaf_elements()
    for leaf_element in leaf_elements:
        leaf_element.add_singular_value_to_approximation()
    return hmat.to_dense_numpy()

def create_hierarchical_test_matrices():    
    res = dict()
    res["name"] = "hierarchical_matrices"

    for idx in range(NB_MATRICES):
        res[idx] = dict()
        mat = get_random_1d_hmat()
        res[idx]["mat"] = mat_to_standard_format(mat)
    
    pickle.dump(res, open("hierarchical_matrices.p", "wb"))

def create_random_toeplitz_matrix():
    column = np.random.uniform(-1, 1, size=(MATRIX_SIZE,))
    row = np.random.uniform(-1, 1, size=(MATRIX_SIZE,))
    row[0] = column[0]

    res = toeplitz(c=column, r=row)
    
    assert_is_matrix(res)
    assert_square(res)
    assert res[0,0] == column[0], "The first entry in the Toeplitz Matrix should match the first entry in the column"
    assert res[1,0] == column[1], "The second entry in the Toeplitz Matrix column should match the second entry in the column"
    assert res[0,1] == row[1], "The second entry in the Toeplith Matrix row should match the second entry in the row"
    assert np.array_equal(np.diag(res),np.diag(res)), "The Toeplitz matrix should have identical elements along its diagonal"
    assert np.array_equal(np.diag(res,1),np.diag(res,1)), "The Toeplitz matrix should have identical elements along its diagonal"
    assert np.array_equal(np.diag(res,-1),np.diag(res,-1)), "The Toeplitz matrix should have identical elements along its diagonal"

    return res.astype(np.float32)

def create_random_cauchy_matrix():
    res = np.zeros(shape=(MATRIX_SIZE, MATRIX_SIZE))
    first_vec = np.random.uniform(-1, 1, size=(MATRIX_SIZE,))
    second_vec = np.random.uniform(-1, 1, size=(MATRIX_SIZE,))

    for row_idx in range(MATRIX_SIZE):
        res[row_idx, :] = 1.0 / (first_vec[row_idx] + second_vec)

    assert not np.any(res == 0), "All values in the original matrix should be replaced to form the cauchy matrix"
    return res.astype(np.float32)

def create_ldr_test_matrices():
    assert NB_MATRICES == 3, "Creating another amount of ldr matrices than 3 is not implemented"
    
    res = dict()
    res["name"] = "ldr_matrices"

    res[0] = dict()
    mat = create_random_toeplitz_matrix().T # Hankel Matrix
    res[0]["mat"] = mat_to_standard_format(mat)

    res[1] = dict()
    mat = create_random_toeplitz_matrix()
    res[1]["mat"] = mat_to_standard_format(mat)

    res[2] = dict()
    mat = create_random_cauchy_matrix()
    res[2]["mat"] = mat_to_standard_format(mat)
    
    pickle.dump(res, open("ldr_matrices.p", "wb"))

if __name__ == "__main__":
    create_sss_test_matrices()
    create_psm_test_matrices()
    create_hierarchical_test_matrices()
    create_ldr_test_matrices()