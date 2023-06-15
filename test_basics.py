import numpy as np
import torch
import pytest
from torch.nn import functional as F
import torch.nn as nn
import math

from npgpt.tensor import Tensor, Linear, Embedding, SequentialModel, Model, Sigmoid, Tanh, ReLU, NewGELU, Softmax, cross_entropy


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
    torch_out = ttorch1 @ ttorch2
    torch_out.backward(torch.ones_like(torch_out))

    # test
    assert np.max(torch_out.detach().numpy()-npt_out.data) < 1e-6
    assert np.max(ttorch1.grad.numpy() - t_ar1.grad) < 1e-6
    assert np.max(ttorch2.grad.numpy() - t_ar2.grad) < 1e-6


def test_linear():
    np.random.seed(35)

    Xi = np.ones((64, 48, 10))
    W = np.ones((10,48))

    npt_Xi = Tensor(Xi)
    npt_W = Tensor(W)
    npt_out = npt_Xi @ npt_W
    npt_out.backward()

    torch_Xi = torch.tensor(Xi, requires_grad=True)
    torch_W = torch.tensor(W.T, requires_grad=True)
    torch_out = F.linear(torch_Xi, torch_W)
    torch_out.backward(torch.ones_like(torch_out))

    assert np.max(np.abs(torch_out.detach().numpy()-npt_out.data)) < 1e-10
    assert np.max(np.abs(torch_Xi.grad.numpy() - npt_Xi.grad)) < 1e-10
    assert np.max(np.abs(torch_W.grad.numpy() - npt_W.grad.T)) < 1e-10


def test_linear_2():
    np.random.seed(35)

    Xi = np.ones((64, 48))
    W = np.ones((48,36))

    npt_Xi = Tensor(Xi)
    npt_W = Tensor(W)
    npt_out = npt_Xi @ npt_W
    npt_out.backward()

    torch_Xi = torch.tensor(Xi, requires_grad=True)
    torch_W = torch.tensor(W.T, requires_grad=True)
    torch_out = F.linear(torch_Xi, torch_W)
    torch_out.backward(torch.ones_like(torch_out))

    assert np.max(np.abs(torch_out.detach().numpy()-npt_out.data)) < 1e-10
    assert np.max(np.abs(torch_Xi.grad.numpy() - npt_Xi.grad)) < 1e-10
    assert np.max(np.abs(torch_W.grad.numpy() - npt_W.grad.T)) < 1e-10


def test_embedding_layer():
    np.random.seed(35)
    vocab_size = 24
    n_embd = 128
    lin_out = 54 

    emb_init_w = np.random.normal(size=(vocab_size, n_embd), scale=0.02)
    lin_init_w = np.random.normal(size=(n_embd, lin_out), scale=0.02)

    idx = np.array([
        [3,2,3,4,1,2,1,1,7,0],
        [3,2,3,4,1,4,1,1,7,0],
        [9,5,4,2,1,3,0,5,12,1],
        [4,2,4,1,6,14,2,5,19,7],
    ])

    # npt
    npt_word_embd = Embedding(vocab_size, n_embd)
    npt_word_embd.weight.data = emb_init_w

    npt_lin = Linear(n_embd, lin_out)
    npt_lin.weight.data = lin_init_w

    npt_out = npt_word_embd(idx)
    npt_out = npt_lin(npt_out)
    npt_out = ReLU(npt_out)

    npt_out.backward()

    # torch
    torch_word_embd = nn.Embedding(vocab_size, n_embd)
    torch_word_embd.weight = torch.nn.Parameter(torch.tensor(emb_init_w, requires_grad=True))

    torch_lin = nn.Linear(n_embd, lin_out, bias=False)
    torch_lin.weight = torch.nn.Parameter(torch.tensor(lin_init_w.T, requires_grad=True))

    torch_out = torch_word_embd(torch.tensor(idx))
    torch_out = torch_lin(torch_out)
    torch_out = nn.ReLU()(torch_out)

    torch_out.backward(torch.ones_like(torch_out))

    assert np.max(np.abs(torch_word_embd.weight.grad.numpy() - npt_word_embd.weight.grad)) < 1e-6


def test_attention_bias():
    np.random.seed(35)

    BLS = 32
    T = 12
    ar1 = np.random.random((8, 4, 12, 6))
    ar2 = np.random.random((8, 4, 6, 12))

    # npt
    t_ar1 = Tensor(ar1)
    t_ar2 = Tensor(ar2)

    # npt_bias = (1-np.tril(np.ones(shape=(1, 1, BLS, BLS)))) #.astype(np.int8)
    # npt_bias[npt_bias==1.] = -np.inf
    npt_bias = (np.tril(np.ones(shape=(1, 1, BLS, BLS))))

    npt_out = t_ar1 @ t_ar2
    # npt_out = npt_out + npt_bias[:,:,:T,:T]
    npt_out.data *= npt_bias[:,:,:T,:T]
    npt_out.data[npt_out.data==0.] = -np.inf

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
    assert np.max(np.abs(ttorch1.grad.numpy() - t_ar1.grad)) < 1e-7
    assert np.max(np.abs(ttorch2.grad.numpy() - t_ar2.grad)) < 1e-7
    assert np.max(np.abs(soft_out_torch.detach().numpy()-soft_npt_out.data)) < 1e-7


def test_simple_softmax():
    np.random.seed(36)
    arr = np.random.random((1, 100))

    npt_arr = Tensor(arr)

    npt_out = npt_arr.softmax()

    npt_out.backward()

    torch_arr = torch.tensor(arr, requires_grad=True)
    torch_out = F.softmax(torch_arr, dim=-1)
    torch_out.backward(torch.ones_like(torch_out))

    assert np.max(np.abs(torch_out.detach().numpy()-npt_out.data)) < 1e-7
    assert np.max(np.abs(torch_arr.grad.numpy() - npt_arr.grad)) < 1e-7


def test_simple_softmax_four_dims():
    np.random.seed(36)
    arr = np.random.random((64, 5, 33, 100))

    npt_arr = Tensor(arr)

    npt_out = npt_arr.softmax()

    npt_out.backward()

    torch_arr = torch.tensor(arr, requires_grad=True)
    torch_out = F.softmax(torch_arr, dim=-1)
    torch_out.backward(torch.ones_like(torch_out))

    assert np.max(np.abs(torch_out.detach().numpy()-npt_out.data)) < 1e-7
    assert np.max(np.abs(torch_arr.grad.numpy() - npt_arr.grad)) < 1e-7


def test_softmax():
    np.random.seed(36)

    ar1 = np.random.random((64, 3, 5, 16))
    ar2 = np.random.random((64, 3, 16, 5))

    # npt
    t_ar1 = Tensor(ar1)
    t_ar2 = Tensor(ar2)

    npt_out = t_ar1 @ t_ar2

    soft_npt_out = npt_out.softmax()
    soft_npt_out.backward()

    # torch
    ttorch1 = torch.tensor(ar1, requires_grad=True)
    ttorch2 = torch.tensor(ar2, requires_grad=True)
    out_torch = torch.matmul(ttorch1, ttorch2)

    soft_out_torch = F.softmax(out_torch, dim=-1)
    soft_out_torch.backward(torch.ones_like(soft_out_torch))

    # test
    assert np.max(np.abs(soft_out_torch.detach().numpy()-soft_npt_out.data)) < 1e-6
    assert np.max(np.abs(ttorch1.grad.numpy() - t_ar1.grad)) < 1e-6
    assert np.max(np.abs(ttorch2.grad.numpy() - t_ar2.grad)) < 1e-6


def test_log_softmax():
    np.random.seed(36)

    arr = np.random.random((16, 64))

    # npt
    t_ar1 = Tensor(arr)

    soft_npt_out = t_ar1.log_softmax()
    soft_npt_out.backward()

    # torch
    ttorch1 = torch.tensor(arr, requires_grad=True)

    soft_out_torch = F.log_softmax(ttorch1, dim=-1)
    soft_out_torch.backward(torch.ones_like(soft_out_torch))

    # test
    assert np.max(np.abs(soft_out_torch.detach().numpy()-soft_npt_out.data)) < 1e-6
    assert np.max(np.abs(ttorch1.grad.numpy() - t_ar1.grad)) < 1e-6


def test_cross_entropy():
    np.random.seed(35)

    BS = 64

    ar1 = np.random.random((BS, 64))
    ar2 = np.random.random((64, 12))
    ar3 = np.random.randint(0, 12, size=BS)

    # npt
    t_ar1 = Tensor(ar1)
    t_ar2 = Tensor(ar2)
    t_ar3 = Tensor(np.eye(12)[ar3])

    npt_out = t_ar1 @ t_ar2
    npt_out = npt_out.reshape(-1, npt_out.shape[-1])
    loss = cross_entropy(npt_out, t_ar3)
    loss.backward()

    # torch
    ttorch1 = torch.tensor(ar1, requires_grad=True)
    ttorch2 = torch.tensor(ar2, requires_grad=True)
    t_ar3 = torch.from_numpy(ar3)
    out_torch = torch.matmul(ttorch1, ttorch2)
    soft_out_torch = F.cross_entropy(out_torch.view(-1, out_torch.size(-1)), t_ar3.view(-1))
    soft_out_torch.backward(torch.ones_like(soft_out_torch))

    # test
    assert np.max(np.abs(ttorch1.grad.numpy() - t_ar1.grad)) < 1e-7
    assert np.max(np.abs(ttorch2.grad.numpy() - t_ar2.grad)) < 1e-7


def test_self_attention():
    np.random.seed(35)
    n_head = 3
    n_embd = 48
    block_size = 6

    BS = 64
    X = np.random.normal(scale=0.01, size=(BS, 5, n_embd)).astype(np.float32)
    attn_W = np.random.random((3*n_embd, n_embd)).astype(np.float32)
    proj_W = np.random.random((n_embd, n_embd)).astype(np.float32)

    # npt
    npt_attn = Linear(n_embd, 3*n_embd)
    npt_attn.weight = Tensor(attn_W.T)
    npt_bias = (1-np.tril(np.ones(shape=(1, 1, block_size, block_size)))) #.astype(np.int8)
    npt_bias[npt_bias==1.] = -np.inf

    npt_out_proj = Linear(n_embd, n_embd)
    npt_out_proj.weight = Tensor(proj_W.T)

    npt_x = Tensor(np.array(X))
    B, T, C = npt_x.shape # batch size, sequence length, embedding dimensionality (n_embd)

    attn_out = npt_attn(npt_x)
    k0, q0, v0 = attn_out.split(3, axis=2)

    npt_k = (1*k0.reshape(B, T, n_head, C // n_head)).transpose(0,2,1,3)  # (B, nh, T, hs)  # reshape directly to correct shape??
    # k = (1*k0).reshape(B, self.n_heads, T, C // self.n_heads) #.transpose(0,2,1,3)  # (B, nh, T, hs)  # reshape directly to correct shape??
    
    npt_q = (1*q0.reshape(B, T, n_head, C // n_head)).transpose(0,2,1,3)  # (B, nh, T, hs)
    # q = (1*q0).reshape(B, self.n_heads, T, C // self.n_heads) #.transpose(0,2,1,3)  # (B, nh, T, hs)  # reshape directly to correct shape??

    npt_v = (1*v0.reshape(B, T, n_head, C // n_head)).transpose(0,2,1,3)  # (B, nh, T, hs)
    # v = (1*v0).reshape(B, self.n_heads, T, C // self.n_heads) #.transpose(0,2,1,3)  # (B, nh, T, hs)  # reshape directly to correct shape??


    npt_att = (npt_q @ npt_k.transpose(0,1,3,2)) / np.sqrt(n_embd) #, nograd=True)

    # apply forward attention mask
    npt_att_b = npt_att + npt_bias[:,:,:T,:T]

    npt_soft_att = Softmax(npt_att_b)

    out = npt_soft_att @ npt_v  # batch, n_heads, seq_len, n_embd // n_heads

    trans_out = 1*out.transpose(0,2,1,3)
    reshape_out = trans_out.reshape(B,T,C)  # B, T, C  # reshape directly to correct shape??

    # reshape_out = out.reshape(B,T,C)  # B, T, C  # reshape directly to correct shape??

    proj_out = npt_out_proj(reshape_out)  

    proj_out.backward()

    # torch
    # init
    c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)  # 3*n_embd, n_embd
    c_attn.weight = torch.nn.Parameter(torch.tensor(attn_W, requires_grad=True))

    c_proj = nn.Linear(n_embd, n_embd, bias=False)  # n_embd, n_embd
    c_proj.weight = torch.nn.Parameter(torch.tensor(proj_W, requires_grad=True))

    bias = torch.tensor(torch.tril(torch.ones(block_size, block_size)), requires_grad=True).view(1, 1, block_size, block_size)
    # exec
    x = torch.tensor(X, requires_grad=True) #.float() #.type(torch.DoubleTensor)
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

    k, q, v  = c_attn(x).split(n_embd, dim=2)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2) # (B, nh, T, hs)
    k.retain_grad()
    v.retain_grad()
    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att0 = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att0.retain_grad()
    att = att0.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
    att.retain_grad()
    soft_att = F.softmax(att, dim=-1)
    soft_att.retain_grad()
    y1 = soft_att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y1.retain_grad()
    yt = y1.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    
    y = c_proj(yt)
    y.backward(torch.ones_like(y))

    assert np.mean(np.abs(y.detach().numpy() - proj_out.data)) < 1e-3
    # assert np.max(np.abs(y.detach().numpy() - proj_out.data)) < 1e-8
