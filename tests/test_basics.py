import numpy as np
import torch
from torch.nn import functional as F

from nptransformer.tensor import Tensor, Linear, SequentialModel, Model, Sigmoid, Tanh, ReLU, NewGELU, Softmax


def test_matmul():
    np.random.seed(35)

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


def test_attention_bias():
    np.random.seed(35)

    BLS = 32
    T = 12
    ar1 = np.random.random((8, 4, 12, 6))
    ar2 = np.random.random((8, 4, 6, 12))

    # npt
    t_ar1 = Tensor(ar1)
    t_ar2 = Tensor(ar2)

    npt_bias = (1-np.tril(np.ones(shape=(1, 1, BLS, BLS)))) #.astype(np.int8)
    npt_bias[npt_bias==1.] = -np.inf

    npt_out = t_ar1 @ t_ar2
    npt_out = npt_out + npt_bias[:,:,:T,:T]
    soft_npt_out = npt_out.softmax()
    soft_npt_out.backward()

    # torch
    t_bias = torch.tensor(torch.tril(torch.ones(BLS, BLS)), requires_grad=False).view(1, 1, BLS, BLS)
    ttorch1 = torch.tensor(ar1, requires_grad=True)
    ttorch2 = torch.tensor(ar2, requires_grad=True)
    out_torch = ttorch1 @ ttorch2
    out_torch = out_torch.masked_fill(t_bias[:,:,:T,:T] == 0, float('-inf'))
    soft_out_torch = F.softmax(out_torch, dim=-1)
    soft_out_torch.backward(torch.ones_like(soft_out_torch))

    # test
    assert np.max(ttorch1.grad.numpy() - t_ar1.grad) < 1e-10
    assert np.max(ttorch2.grad.numpy() - t_ar2.grad) < 1e-10
    assert np.max(soft_out_torch.detach().numpy()-soft_npt_out.data) < 1e-10


def test_softmax():
    np.random.seed(35)

    ar1 = np.random.random((8))
    ar2 = np.random.random((8, 12))

    # npt
    t_ar1 = Tensor(ar1)
    t_ar2 = Tensor(ar2)

    npt_out = t_ar1.dot(t_ar2)
    soft_npt_out = npt_out.softmax()
    soft_npt_out.backward()

    # torch
    ttorch1 = torch.tensor(ar1, requires_grad=True)
    ttorch2 = torch.tensor(ar2, requires_grad=True)
    out_torch = torch.matmul(ttorch1, ttorch2)
    soft_out_torch = F.softmax(out_torch, dim=-1)
    soft_out_torch.backward(torch.ones_like(soft_out_torch))

    # test
    assert np.max(soft_out_torch.detach().numpy()-soft_npt_out.data) < 1e-10
    assert np.max(ttorch1.grad.numpy() - t_ar1.grad) < 1e-10
    assert np.max(ttorch2.grad.numpy() - t_ar2.grad) < 1e-10


def test_cross_entropy():
    np.random.seed(35)

    ar1 = np.random.random((80))
    ar2 = np.random.random((80, 12))
    ar3 = np.zeros(12)
    ar3[0] = 1

    # npt
    t_ar1 = Tensor(ar1)
    t_ar2 = Tensor(ar2)
    t_ar3 = Tensor(ar3)

    npt_out = t_ar1.dot(t_ar2)
    soft_npt_out = npt_out.softmax()
    loss = -(soft_npt_out.log()*t_ar3 + (1-t_ar3)*(1-soft_npt_out).log())
    loss.backward()

    # torch
    ttorch1 = torch.tensor(ar1, requires_grad=True)
    ttorch2 = torch.tensor(ar2, requires_grad=True)
    t_ar3 = torch.from_numpy(ar3)
    out_torch = torch.matmul(ttorch1, ttorch2)
    soft_out_torch = F.cross_entropy(out_torch, t_ar3)
    soft_out_torch.backward(torch.ones_like(soft_out_torch))

    # test
    assert np.max(ttorch1.grad.numpy() - t_ar1.grad) < 1e-8
    assert np.max(ttorch2.grad.numpy() - t_ar2.grad) < 1e-8


if __name__=='__main__':
    test_attention_bias()