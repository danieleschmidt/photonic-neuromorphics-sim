"""
Mock PyTorch implementation for testing purposes.
"""

import numpy as np
from typing import Union, Tuple, List, Any, Optional


class MockTensor:
    """Mock PyTorch tensor for testing."""
    
    def __init__(self, data):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        elif isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __getitem__(self, key):
        return MockTensor(self.data[key])
    
    def __setitem__(self, key, value):
        if isinstance(value, MockTensor):
            self.data[key] = value.data
        else:
            self.data[key] = value
    
    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data + other.data)
        return MockTensor(self.data + other)
    
    def __sub__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data - other.data)
        return MockTensor(self.data - other)
    
    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data * other.data)
        return MockTensor(self.data * other)
    
    def __truediv__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data / other.data)
        return MockTensor(self.data / other)
    
    def __gt__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data > other.data)
        return MockTensor(self.data > other)
    
    def __lt__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data < other.data)
        return MockTensor(self.data < other)
    
    def sum(self, dim=None):
        if dim is None:
            return MockTensor(np.sum(self.data))
        return MockTensor(np.sum(self.data, axis=dim))
    
    def mean(self, dim=None):
        if dim is None:
            return MockTensor(np.mean(self.data))
        return MockTensor(np.mean(self.data, axis=dim))
    
    def std(self, dim=None):
        if dim is None:
            return MockTensor(np.std(self.data))
        return MockTensor(np.std(self.data, axis=dim))
    
    def float(self):
        return MockTensor(self.data.astype(np.float32))
    
    def clone(self):
        return MockTensor(self.data.copy())
    
    def item(self):
        return self.data.item()
    
    def numpy(self):
        return self.data
    
    def dim(self):
        return len(self.data.shape)
    
    def nelement(self):
        return self.data.size
    
    def element_size(self):
        return self.data.itemsize
    
    def __repr__(self):
        return f"MockTensor({self.data})"


def tensor(data, dtype=None):
    """Create a mock tensor."""
    return MockTensor(data)


def zeros(*shape):
    """Create tensor filled with zeros."""
    return MockTensor(np.zeros(shape))


def ones(*shape):
    """Create tensor filled with ones.""" 
    return MockTensor(np.ones(shape))


def rand(*shape):
    """Create tensor with random values."""
    return MockTensor(np.random.rand(*shape))


def randn(*shape):
    """Create tensor with random normal values."""
    return MockTensor(np.random.randn(*shape))


def stack(tensors, dim=0):
    """Stack tensors along dimension."""
    arrays = [t.data if isinstance(t, MockTensor) else t for t in tensors]
    return MockTensor(np.stack(arrays, axis=dim))


def cat(tensors, dim=0):
    """Concatenate tensors along dimension."""
    arrays = [t.data if isinstance(t, MockTensor) else t for t in tensors]
    return MockTensor(np.concatenate(arrays, axis=dim))


def corrcoef(tensors):
    """Calculate correlation coefficient."""
    arrays = [t.data if isinstance(t, MockTensor) else t for t in tensors]
    return MockTensor(np.corrcoef(arrays))


def argmax(tensor, dim=None):
    """Return argmax along dimension."""
    if isinstance(tensor, MockTensor):
        return MockTensor(np.argmax(tensor.data, axis=dim))
    return MockTensor(np.argmax(tensor, axis=dim))


def allclose(a, b, rtol=1e-5, atol=1e-8):
    """Check if tensors are element-wise equal within tolerance."""
    a_data = a.data if isinstance(a, MockTensor) else a
    b_data = b.data if isinstance(b, MockTensor) else b
    return np.allclose(a_data, b_data, rtol=rtol, atol=atol)


def any(tensor):
    """Check if any element is True."""
    data = tensor.data if isinstance(tensor, MockTensor) else tensor
    return bool(np.any(data))


def logical_or(a, b):
    """Element-wise logical OR."""
    a_data = a.data if isinstance(a, MockTensor) else a
    b_data = b.data if isinstance(b, MockTensor) else b
    return MockTensor(np.logical_or(a_data, b_data))


def isnan(tensor):
    """Check for NaN values."""
    data = tensor.data if isinstance(tensor, MockTensor) else tensor
    return MockTensor(np.isnan(data))


def isinf(tensor):
    """Check for infinite values."""
    data = tensor.data if isinstance(tensor, MockTensor) else tensor
    return MockTensor(np.isinf(data))


def sin(tensor):
    """Element-wise sine."""
    data = tensor.data if isinstance(tensor, MockTensor) else tensor
    return MockTensor(np.sin(data))


def linspace(start, end, steps):
    """Create linearly spaced tensor."""
    return MockTensor(np.linspace(start, end, steps))


class MockFFT:
    """Mock FFT operations."""
    
    @staticmethod
    def fft(tensor, dim=None):
        data = tensor.data if isinstance(tensor, MockTensor) else tensor
        result = np.fft.fft(data, axis=dim)
        return MockTensor(result)
    
    @staticmethod
    def ifft(tensor, dim=None):
        data = tensor.data if isinstance(tensor, MockTensor) else tensor
        result = np.fft.ifft(data, axis=dim)
        return MockTensor(result)


fft = MockFFT()


class MockNN:
    """Mock neural network module."""
    
    class Module:
        def __init__(self):
            pass
        
        def eval(self):
            return self
        
        def train(self):
            return self
        
        def parameters(self):
            return []
        
        def named_parameters(self):
            return []
        
        def __call__(self, x):
            return self.forward(x)
        
        def forward(self, x):
            return x
    
    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = MockTensor(np.random.randn(out_features, in_features) * 0.1)
            self.bias = MockTensor(np.zeros(out_features)) if bias else None
        
        def forward(self, x):
            result = MockTensor(x.data @ self.weight.data.T)
            if self.bias is not None:
                result = result + self.bias
            return result
    
    class Sequential(Module):
        def __init__(self, *layers):
            super().__init__()
            self.layers = layers
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
        
        def __iter__(self):
            return iter(self.layers)
    
    class ReLU(Module):
        def forward(self, x):
            return MockTensor(np.maximum(0, x.data))
    
    class MSELoss(Module):
        def forward(self, input, target):
            input_data = input.data if isinstance(input, MockTensor) else input
            target_data = target.data if isinstance(target, MockTensor) else target
            return MockTensor(np.mean((input_data - target_data) ** 2))
        
        def __call__(self, input, target):
            return self.forward(input, target)
    
    class LSTM(Module):
        def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.batch_first = batch_first
        
        def forward(self, x):
            # Simplified LSTM - just return input with modified shape
            if self.batch_first:
                batch_size, seq_len = x.shape[:2]
                output_shape = (batch_size, seq_len, self.hidden_size)
            else:
                seq_len, batch_size = x.shape[:2]
                output_shape = (seq_len, batch_size, self.hidden_size)
            
            output = MockTensor(np.random.randn(*output_shape))
            hidden = (MockTensor(np.zeros((self.num_layers, batch_size, self.hidden_size))),
                     MockTensor(np.zeros((self.num_layers, batch_size, self.hidden_size))))
            
            return output, hidden
    
    class Dropout(Module):
        def __init__(self, p=0.5):
            super().__init__()
            self.p = p
        
        def forward(self, x):
            return x  # Simplified - no actual dropout


nn = MockNN()


# Create a mock torch module
class MockTorch:
    def __init__(self):
        self.nn = nn
        self.fft = fft
        
        # Add all functions as attributes
        self.tensor = tensor
        self.zeros = zeros
        self.ones = ones
        self.rand = rand
        self.randn = randn
        self.stack = stack
        self.cat = cat
        self.corrcoef = corrcoef
        self.argmax = argmax
        self.allclose = allclose
        self.any = any
        self.logical_or = logical_or
        self.isnan = isnan
        self.isinf = isinf
        self.sin = sin
        self.linspace = linspace
        
        # Tensor type
        self.Tensor = MockTensor
        self.FloatTensor = MockTensor


# Export the mock torch
torch = MockTorch()