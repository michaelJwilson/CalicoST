import pytest
import numpy as np
from scipy.sparse import csr_matrix
from calicost.hmrf import solve_edges

@pytest.fixture
def mock_data():
    ii = 4
    n_clones = 5

    adjacency_mat = np.random.randint(0, 10, size=(10,10))
    new_assignment = np.random.randint(0, n_clones, size=10)

    return ii, n_clones, adjacency_mat, new_assignment
    
def test_edges_old(benchmark, mock_data):
    ii, n_clones, adjacency_mat, new_assignment = mock_data
    
    def get_exp():
        return solve_edges(ii, csr_matrix(adjacency_mat), new_assignment, n_clones, new=False)
    
    exp = benchmark(get_exp)

def test_edges_new(benchmark, mock_data):
    ii, n_clones, adjacency_mat, new_assignment = mock_data

    def get_result():
        return solve_edges(ii, csr_matrix(adjacency_mat), new_assignment, n_clones, new=True)

    result = benchmark(get_result)

def test_edges_equality(mock_data):
    ii, n_clones, adjacency_mat, new_assignment = mock_data

    exp = solve_edges(ii, csr_matrix(adjacency_mat), new_assignment, n_clones, new=False)
    result = solve_edges(ii, csr_matrix(adjacency_mat), new_assignment, n_clones, new=True)
    
    assert np.all(exp == result)
