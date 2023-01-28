import numpy as np
import torch
from nptransformer.tensor import Tensor, Linear, SequentialModel, Model, Sigmoid, Tanh, ReLU, NewGELU, Softmax


np.random.seed(35)


def test_matmul():
    ar1 = np.random.random((8, 4, 6, 12))
    ar2 = np.random.random((8, 4, 12, 6))

    # npt
    t_ar1 = Tensor(ar1)
    t_ar2 = Tensor(ar2)
    npt_out = t_ar1 @ t_ar2
    npt_out.backward()

    # torch
    ttorch1 = torch.tensor(ar1, requires_grad=True)
    ttorch2 = torch.tensor(ar2, requires_grad=True)
    out_torch = ttorch1 @ ttorch2
    out_torch.backward(torch.ones_like(out_torch))

    # test
    assert np.max(out_torch.detach().numpy()-npt_out.data) < 1e-10
    assert np.max(ttorch1.grad.numpy() - t_ar1.grad) < 1e-10
    assert np.max(ttorch2.grad.numpy() - t_ar2.grad) < 1e-10
