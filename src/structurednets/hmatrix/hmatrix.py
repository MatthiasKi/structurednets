import torch

from structurednets.hmatrix.block_cluster_tree import BlockClusterTree

# NOTE: For the approximation, just use HODLR - the level of how many divisions can be seen as a hyper parameter
# Within the division, I can use the SVD and iteratively select the singular value which should be added in order to improve the approximation capability the most

class HMatrix:
    def __init__(self, block_cluster_tree: BlockClusterTree):
        self.block_cluster_tree = block_cluster_tree

    def to_dense() -> torch.tensor:
        pass

    def dot(vec: torch.tensor) -> torch.tensor:
        assert len(vec.shape) == 1 or vec.shape[0] == 1 or vec.shape[1] == 1, "dot is only implemented for vectors"
        pass