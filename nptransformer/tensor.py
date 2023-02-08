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

def LogSoftmax(data: 'Tensor'):
    return data.log_softmax()

def Exp(data: 'Tensor'):
    return data.exp()

class Model:

    def __init__(self):
        self.training = True

    @abstractmethod
    def forward(self, x: 'Tensor'):
        pass

    def __call__(self, x: 'Tensor'):
        return self.forward(x)

    def zero_grad(self):
        for tensor in self.get_trainable_tensors():
            tensor.grad = np.zeros_like(tensor.data)

    def get_trainable_tensors(self):
        trainable_tensors = set()
        for att in dir(self):
            attribute = getattr(self, att)
            if isinstance(attribute, Tensor) and attribute._nograd is False:
                trainable_tensors.add(attribute)
            elif issubclass(type(attribute), Model):
                trainable_tensors |= attribute.get_trainable_tensors()
            elif isinstance(attribute, list):
                for el in attribute:
                    if issubclass(type(el), Model):
                        trainable_tensors |= el.get_trainable_tensors()
                    elif isinstance(el, Tensor) and el._nograd is False:
                        trainable_tensors.add(el)
        return trainable_tensors

    def get_model_objects(self):
        model_objects = set({self})
        for att in dir(self):
            attribute = getattr(self, att)
            if issubclass(type(attribute), Model):
                model_objects |= attribute.get_model_objects()
            elif isinstance(attribute, list):
                for el in attribute:
                    if issubclass(type(el), Model):
                        model_objects |= el.get_model_objects()

        return model_objects

    def train(self):
        """Set to train mode."""
        for model_obj in self.get_model_objects():
            model_obj.training = True

    def eval(self):
        """Set to eval mode."""
        for model_obj in self.get_model_objects():
            model_obj.training = False

class Dropout(Model):

    def __init__(self, p=0.1):
        """
        :param p: probability of an element being zeroed.
        """
        super().__init__()
        self.p = p

    def forward(self, x: 'Tensor'):
        if self.training:
            x *= 1/(1-self.p)
            mask = (np.random.rand(*x.shape) > self.p)
            x *= mask # + 1e-8
        return x


class SequentialModel(Model):
    """
    Simple sequential model
    """

    def __init__(self, layers: list):
        super().__init__()
        self.layers = layers

    def forward(self, x: 'Tensor'):
        for layer in self.layers:
            x = layer(x)
        return x


class Embedding(Model):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(np.random.normal(size=(num_embeddings, embedding_dim), scale=0.02), name="Embedding", nograd=False)

    def forward(self, vocab_ids):

        one_hot = Tensor(np.eye(self.num_embeddings)[vocab_ids], nograd=True)

        return one_hot @ self.weight


class Linear(Model):

    def __init__(self, nin, nout, nonlin=None, use_bias=False, name='') -> None:
        super().__init__()
        self.nonlin = nonlin
        self.use_bias = use_bias

        scale = 0.02  #
        # scale = np.sqrt(2/(nin+nout))

        npw = np.random.normal(size=(nin, nout), scale=scale)
        self.weight = Tensor(npw, name=name)

        self.B = None
        if self.use_bias:
            self.B = Tensor(np.zeros(nout))

    def __call__(self, x) -> 'Tensor':
        return self.forward(x)
        
    def forward(self, x) -> 'Tensor':
        x = x @ self.weight
        if self.use_bias:
            x += self.B
        if self.nonlin:
            x = self.nonlin(x)

        return x


class Tensor:

    def __init__(self, data, _children=(), _op='', nograd=False, grad=None, name=''):
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._nograd = nograd
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

        self.name = f"{name}-[_op:{_op}]"
        self.data = np.array(data).astype(np.float32)
        self.grad = None
        self.grad_mask = None
        if not nograd:
            self.grad = np.zeros(shape=self.data.shape).astype(np.float32) if grad is None else grad
            if grad is not None:
                assert grad.shape == self.data.shape

    def __repr__(self):
        return f"Tensor({self.shape})[{self.name}]: {self.data}"

    def __add__(self, other):
        # other = other if isinstance(other, Tensor) else Tensor(other)
        if not isinstance(other, Tensor):
            other = Tensor(other, nograd=True)

        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            if not self._nograd:
                self.grad += out.grad
            if not other._nograd:
                if other.grad.shape[-1] == out.grad.shape[-1] and other.grad.shape != out.grad.shape:  # Bias
                    other.grad += out.grad.reshape(np.prod(out.grad.shape[:-1]), out.grad.shape[-1]).sum(axis=0)
                else:
                    other.grad += out.grad
        out._backward = _backward

        return out

    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data), (self, other), 'dot')

        def _backward():
            if not self._nograd:
                self.grad += out.grad.dot(other.data.swapaxes(-1,-2))
            if not other._nograd:
                other.grad += self.data.T.reshape(self.data.T.shape[0], int(np.prod(self.data.T.shape[1:]))).dot(
                                                                (out.grad.reshape(int(np.prod(out.grad.shape[:-1])), out.grad.shape[-1]))
                                                            )

        out._backward = _backward

        return out

    def split(self, indices_or_sections, axis=0):
        new_tensors = [
            Tensor(arr, grad=grad, _children=(self,), _op='split') 
            for arr, grad in zip(
                np.split(self.data, indices_or_sections=indices_or_sections, axis=axis),
                np.split(self.grad, indices_or_sections=indices_or_sections, axis=axis),
            )
        ]

        return new_tensors

    @property
    def T(self):
        assert len(self.shape) == 2
        out = Tensor(self.data.T, grad=self.grad.T, _children=(self,), _op='T')
        return out

    def transpose(self, *args):
        out = Tensor(self.data.transpose(*args), grad=self.grad.transpose(*args), _children=(self,), _op='transpose')
        return out

    def reshape(self, *args):
        out = Tensor(self.data.reshape(*args), grad=self.grad.reshape(*args), _children=(self,), _op='reshape')
        return out        

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, nograd=True)

        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            if not self._nograd:
                self.grad += other.data * out.grad
            if not other._nograd:
                # logger.info(f"{self.data=} ; {out.data=}")
                other.grad += self.data * out.grad 
        out._backward = _backward

        return out
    
    def __matmul__(self, other):  # @ operator
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            if not self._nograd:
                self.grad += (other.data @ out.grad.swapaxes(-1,-2)).swapaxes(-1,-2)
            if not other._nograd:
                rev_grad = self.data.swapaxes(-1,-2) @ out.grad
                if len(rev_grad.shape) > len(other.grad.shape):
                    other.grad += np.sum(rev_grad, axis=0)
                else:
                    other.grad += rev_grad
                # other.grad += np.sum(self.data.swapaxes(-1,-2) @ out.grad, axis=0)  # hmm
                # other.grad += self.data.swapaxes(-1,-2) @ out.grad

        out._backward = _backward

        return out

    def __rmatmul__(self, other):
        return other @ self

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

    def log_softmax(self):
        self.data -= np.max(self.data, axis=-1, keepdims=True) 
        exp_self = np.exp(self.data)
        softmax_self = exp_self / np.sum(exp_self, axis=-1, keepdims=True)

        out = Tensor(self.data-np.log(np.sum(exp_self, axis=-1, keepdims=True)), (self,), 'log_softmax')

        def _backward():
            # self.grad += (1-softmax_self) * out.grad
            # self.grad += (1-np.sum(softmax_self*out.grad, axis=-1, keepdims=True))
            # self.grad += out.grad - (np.exp(out.data).T * np.sum(out.grad, axis=-1)).T
            self.grad += out.grad - (softmax_self.T * np.sum(out.grad, axis=-1)).T

        out._backward = _backward
        return out

    def softmax(self):
        self.data -= np.max(self.data, axis=-1, keepdims=True)  # stability 

        exp_self = np.exp(self.data)
        out = Tensor(exp_self / np.sum(exp_self, axis=-1, keepdims=True), (self,), 'softmax')

        def _backward():
            self.grad += (out.data * (out.grad.transpose((-1, *tuple(range(len(out.grad.shape)-1)))) - np.sum(out.grad*out.data, axis=-1)).transpose(((*tuple(range(1, len(out.grad.shape))), 0))))  #?
            # self.grad += (out.data * (out.grad - out.grad*out.data))
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
        if self.grad_mask is not None:
            self.grad = (self.grad.T*self.grad_mask).T
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

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def sum(self, axis=None):
        # out = Tensor(np.sum(self.data, axis=axis), _children=(self))
        # def _backward():
        #     self.grad += out.grad
        # out._backward = _backward

        # return out
        return np.sum(self.data, axis=axis)

    def mean(self, axis=None):
        return np.mean(self.data, axis=axis)
