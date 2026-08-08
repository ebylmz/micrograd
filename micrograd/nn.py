import random

from micrograd.engine import Value


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool = True) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x: list[Value | float | int]) -> Value:
        # act = w*x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin: int, nout: int, nonlin: bool) -> None:
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]

    def __call__(self, x: list[Value | float | int]) -> Value | list[Value]:
        x = [x] if not isinstance(x, list) else x 
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]) -> None:
        nlayers = len(nouts)
        nins = [nin] + nouts
        self.layers = [Layer(nins[i], nouts[i], nonlin=(i+1)!=nlayers) for i in range(nlayers)]

    def __call__(self, x: list[Value | float | int]) -> Value | list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x        

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]
