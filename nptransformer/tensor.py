"""
Adapted from https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py
"""

import numpy as np
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def ReLU(data: 'Tensor'):
    return data.relu()

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

    def __init__(self, nin, nout, nonlin=False) -> None:

        self.nonlin = nonlin

        scale = np.sqrt(2/(nin+nout))
        npw = np.random.normal(size=(nin, nout), scale=scale)
        
        self.W = Tensor(npw)

    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        x = x.dot(self.W)
        if self.nonlin:
            x = ReLU(x)
        return x


class Tensor:

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = np.zeros(shape=data.shape)
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
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

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
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

