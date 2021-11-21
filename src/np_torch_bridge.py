import numpy as np
import torch


def _numpy_to_torch():
    np_array = np.ones((2, 2))
    print(np_array)
    print(type(np_array))

    torch_tensor = torch.from_numpy(np_array)
    print(torch_tensor)
    print(type(torch_tensor))


def _data_type():
    # Data types matter: intentional error
    np_array_new = np.ones((2, 2), dtype=np.int8)
    torch.from_numpy(np_array_new)
    """**The conversion supports:**
    1. `double`
    2. `float` 
    3. `int64`, `int32`, `uint8` 
    """
    # Data types matter
    np_array_new = np.ones((2, 2), dtype=np.int64)
    torch.from_numpy(np_array_new)

    # Data types matter
    np_array_new = np.ones((2, 2), dtype=np.int32)
    torch.from_numpy(np_array_new)


def _torch_to_numpy():
    torch_tensor = torch.ones(2, 2)
    type(torch_tensor)
    torch_to_numpy = torch_tensor.numpy()
    type(torch_to_numpy)


if __name__ == "__main__":
    _numpy_to_torch()
    _data_type()
    _torch_to_numpy()
