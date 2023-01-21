"""
Adapted from https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py
"""

from abc import abstractmethod
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

def Exp(data: 'Tensor'):
    return data.exp()

class Model:

    @abstractmethod
    def forward(self, x: 'Tensor'):
        pass

    def __call__(self, x: 'Tensor'):
        return self.forward(x)


class SequentialModel(Model):
    """
    Simple sequential model
    """

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: 'Tensor'):
        for layer in self.layers:
            x = layer(x)
        return x

# class Switch:

#     def __init__(self, lays: list):
#         self.
#         self.W = Tensor()  #...

#     self forward(self, x):
#         # ...

#     self backward(self):


class Embedding(Model):

    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.W = Tensor(np.random.normal(size=(num_embeddings, embedding_dim)))

    def forward(self, vocab_ids):

        one_hot = Tensor(np.eye(self.num_embeddings)[vocab_ids], nograd=True)

        return one_hot.dot(self.W)  #split(self.num_embeddings) #[vocab_id]


class TestModel(Model):
    def __init__(self):
        self.fc1 = Linear(784, 64, nonlin=ReLU)

        self.fc2a = Linear(32, 32, nonlin=ReLU)
        self.fc2b = Linear(32, 32, nonlin=ReLU)
        # self.fc2a = FC(16, 16, nonlin=ReLU)
        # self.fc2b = FC(16, 16, nonlin=ReLU)
        # self.fc2c = FC(16, 16, nonlin=ReLU)
        # self.fc2d = FC(16, 16, nonlin=ReLU)

        self.fc3 = Linear(64, 10)

    def forward(self, x: 'Tensor'):

        x = self.fc1(x)
        x1, x2 = x.split(2, -1)
        x1 = self.fc2a(x1)
        x2 = self.fc2b(x2)
        # x = ...

        x = self.fc3(x)

        return x


class Linear:

    def __init__(self, nin, nout, nonlin=None) -> None:

        self.nonlin = nonlin

        scale = np.sqrt(2/(nin+nout))
        npw = np.random.normal(size=(nin, nout), scale=scale)
        
        self.W = Tensor(npw)

    def __call__(self, x) -> 'Tensor':
        return self.forward(x)
        
    def forward(self, x) -> 'Tensor':
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
            if not self._nograd:
                self.grad += out.grad.dot(other.data.T)
            if not other._nograd:
                other.grad += self.data.T.dot(out.grad)

        out._backward = _backward

        return out

    def split(self, indices_or_sections, axis=0):
        new_tensors = [
            Tensor(arr, grad=grad, _children=(self,)) 
            for arr, grad in zip(
                np.split(self.data, indices_or_sections=indices_or_sections, axis=axis),
                np.split(self.grad, indices_or_sections=indices_or_sections, axis=axis),
            )
        ]

        return new_tensors

    @property
    def T(self):
        assert len(self.shape) == 2
        out = Tensor(self.data.T, grad=self.grad.T, children=(self,))
        return out

    def transpose(self, *args):
        out = Tensor(self.data.transpose(*args), grad=self.grad.transpose(*args), children=(self,))
        return out

    def reshape(self, *args):
        out = Tensor(self.data.reshape(*args), grad=self.grad.reshape(*args), children=(self,))
        return out

    # @classmethod
    # def concat(tensors, axis=-1):

    #     tensor_data = np.concatenate([tensor.data for tensor in tensors], axis=axis)
    #     grad = np.zeros_like(tensor_data)
    #     # for i, g in enumerate(grad.split(grad, indices_or_sections=len(tensors), axis=axis)):
    #     #     tensors[i] 
    #     out = Tensor( , op='concat')

    #     return out
        

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

    def __matmul___(self, other):  # @ operator
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            # pass
            pass
        out._backward = _backward

        return out

    def __rmatmul__(self, other):
        raise NotImplementedError

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
        out = 1/(1+Exp(-self))
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        eps = 1e-8
        out = Tensor(np.log(self.data + eps), (self,), 'log')

        def _backward():
            self.grad += 1/(self.data + eps) * out.grad
        out._backward = _backward

        return out

    def softmax(self):
        self.data -= np.max(self.data, axis=-1, keepdims=True)  # stability 

        exp_self = np.exp(self.data)
        out = Tensor(exp_self / np.sum(exp_self, axis=-1, keepdims=True), (self,), 'softmax')

        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad
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

