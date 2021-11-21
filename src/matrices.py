import numpy as np
import torch


def _create_matrices():
    # Creating a 2x2 array
    arr = [[1, 2], [3, 4]]
    # Convert to NumPy
    np.array(arr)
    # Convert to PyTorch Tensor
    torch.Tensor(arr)


def _create_matrices_default_val():
    # All the following commands should print out the same result
    np.ones((2, 2))
    torch.ones((2, 2))
    np.random.rand(2, 2)
    torch.rand(2, 2)


def _np_random_with_seeds():
    # Seed
    np.random.seed(0)
    np.random.rand(2, 2)
    # Seed
    np.random.seed(0)
    np.random.rand(2, 2)
    # No seed
    np.random.rand(2, 2)
    # No seed
    np.random.rand(2, 2)


def _torch_random_with_seeds():
    # Torch Seed
    torch.manual_seed(0)
    torch.rand(2, 2)
    # Torch Seed
    torch.manual_seed(0)
    torch.rand(2, 2)
    # Torch No Seed
    torch.rand(2, 2)
    # Torch No Seed
    torch.rand(2, 2)


if __name__ == "__main__":
    """
    This file covers some basics for np and torch matrices
    """
    _create_matrices()
    _create_matrices_default_val()
    _np_random_with_seeds()
    _torch_random_with_seeds()
