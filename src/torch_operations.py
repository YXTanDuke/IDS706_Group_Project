import numpy as np
import torch


def _basic_ops():
    a = torch.ones(2, 2)
    print(a)
    print(a.size())

    a = torch.ones(2, 2)
    b = torch.ones(2, 2)

    print("Additions")
    c = a + b
    print(c)
    c = torch.add(a, b)
    print(c)

    print("Subtractions")
    print(a - b)
    # Not in-place
    print(a.sub(b))
    print(a)
    # Inplace
    print(a.sub_(b))
    print(a)


    print("Multiplications")
    a = torch.ones(2, 2)
    b = torch.zeros(2, 2)
    print(a * b)
    # Not in-place
    print(torch.mul(a, b))
    print(a)
    # In-place
    print(a.mul_(b))
    print(a)

    
    print("Divisions")
    a = torch.ones(2, 2)
    b = torch.zeros(2, 2)
    print(b / a)
    print(torch.div(b, a))
    # Inplace
    print(b.div_(a))


def _calculate_mean():
    print("Mean")
    a = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(a.size())
    print(a.mean(dim=0))
    print(a.mean(dim=1))


def _calculate_std():
    a = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(a.std(dim=0))


if __name__ == "__main__":
    _basic_ops()
    _calculate_mean()
    _calculate_std()