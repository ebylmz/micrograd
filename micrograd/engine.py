import math

class Value:
    """ Stores a single scalar value and its gradient """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._children = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.label = label
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,  (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')        

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self, ), f"**{other}")

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0.0 else self.data, (self, ), 'ReLU')

        def _backward():
            # local derivative is 0.0 or 1.0
            self.grad += (out.data > 0.0) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        # tanh = e**(2x) - 1 / e**(2x) + 1
        e = math.exp(2*self.data)
        t = (e - 1) / (e + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            # derivative of tanh is 1 - tanh(n)**2
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            # Build topological sort using DFS post-order traversal
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        # Make the topological order from root to leaf nodes    
        build_topo(self)
        # Set the base case
        self.grad = 1.0       
        # Call backward
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self 
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __truediv__(self, other): # self / other
        return self * other**-1

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"