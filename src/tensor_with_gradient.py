import numpy as np
import torch


def _get_tensor_with_gradient():
    a = torch.ones((2, 2), requires_grad=True)
    a.requires_grad

    # Behaves similarly to tensors
    b = torch.ones((2, 2), requires_grad=True)
    print(a + b)
    print(torch.add(a, b))

    print(a * b)
    print(torch.mul(a, b))

    x = torch.ones(2, requires_grad=True)
    y = 5 * (x + 1) ** 2
    o = (1/2) * torch.sum(y)
    o.backward()
    print(x.grad)


if __name__ == "__main__":
    _get_tensor_with_gradient()
