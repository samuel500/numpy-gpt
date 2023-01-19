"""
Adapted from https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py
"""

import numpy as np
import logging
import math
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def ReLU(data: 'Tensor'):
    return data.relu()

def NewGELU(data: 'Tensor'):
    return data.gelu()

def Tanh(data: 'Tensor'):
    return data.tanh()

def Sigmoid(data: 'Tensor'):
    return data.sigmoid()

def Softmax(data: 'Tensor'):
    return data.softmax()

class SequentialModel:
    """
    Simple sequential model
    """

    def __init__(self, layers: list):

        self.layers = layers

    def __call__(self, x: 'Tensor'):
        return self.forward(x)

    def forward(self, x: 'Tensor'):
        for layer in self.layers:
            x = layer(x)
        return x


class FC:

    def __init__(self, nin, nout, nonlin=None) -> None:

        self.nonlin = nonlin

        scale = np.sqrt(2/(nin+nout))
        npw = np.random.normal(size=(nin, nout), scale=scale)
        
        self.W = Tensor(npw)

    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        x = x.dot(self.W)
        if self.nonlin:
            x = self.nonlin(x)

        return x


class Tensor:

    def __init__(self, data, _children=(), _op='', nograd=False, grad=None):
        self.data = np.array(data) 
        self.grad = np.zeros(shape=self.data.shape) if grad is None else grad
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._nograd = nograd
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        # other = other if isinstance(other, Tensor) else Tensor(other)
        if not isinstance(other, Tensor):
            other = Tensor(other, nograd=True)

        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            if not self._nograd:
                self.grad += out.grad
            if not other._nograd:
                other.grad += out.grad
        out._backward = _backward

        return out

    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data), (self, other), 'dot')

        def _backward():
            # logger.info(f"{other.grad.shape=}, {self.data.shape=}, {out.grad.shape=}")
            other.grad += self.data.T.dot(out.grad) # out.grad.dot(other.data.T) 
            # logger.info(f"{self.grad.shape=}, {other.grad.shape=}, {other.data.shape=}")
            self.grad += out.grad.dot(other.data.T)
        out._backward = _backward

        return out

    def split(self, indices_or_sections, axis=0):
        new_tensors = [
            Tensor(arr, grad=grad) 
            for arr, grad in zip(
                np.split(self.data, indices_or_sections=indices_or_sections, axis=axis),
                np.split(self.grad, indices_or_sections=indices_or_sections, axis=axis),
            )
        ]

        return new_tensors

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, nograd=True)

        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            if not self._nograd:
                self.grad += other.data * out.grad
            if not other._nograd:
                other.grad += self.data * out.grad 
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def gelu(self):
        out = 0.5 * self * (1.0 + Tanh(math.sqrt(2.0/math.pi) * (self + 0.0044715 * self**3)))
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), (self,), 'Tanh')

        def _backward():
            self.grad += (1-out.data**2) * out.grad
        out._backward = _backward

        return out

    def sqrt(self):
        out = self**0.5
        return out

    def sigmoid(self):
        out = Tensor(1/(1+np.exp(-self.data)))

        def _backward():
            self.grad += (1-out.data)*out.data * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Tensor(np.exp(self.data))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def softmax(self):
        out = self.exp()
        out /= Tensor(np.sum(out, axis=1, keepdims=True), nograd=True)
        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones(self.data.shape)
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * Tensor(-1*np.ones(self.shape))

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

