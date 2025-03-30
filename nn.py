import random
from autograd import Value, Relu
from typing import List

class Neuron:
    def __init__(self, input_len: int, activation_fn=Relu()):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(input_len)]
        self.b = Value(random.uniform(-1, 1))
        self.activation_fn = activation_fn
    
    def __call__(self, inputs: List[Value]):
        activation = sum(input * weight for input, weight in zip(inputs, self.w)) + self.b
        return self.activation_fn(activation)

class Layer:
    def __init__(self, input_len: int, n_neurons: int, activation_fn=Relu()):
        self.neurons = [Neuron(input_len, activation_fn) for _ in range(n_neurons)]
        self.input_len = input_len
        self.output_len = n_neurons

    def __call__(self, inputs: List[Value]):
        return [neuron(inputs) for neuron in self.neurons]

class MLP:
    def __init__(self):
        self.layers = []

    def __init__(self, input_len: int, output_lens: List[int], activation_fn=Relu()):
        self.layers = [Layer(input_len, output_lens[0], activation_fn)]
        for i in range(1, len(output_lens)):
            self.layers.append(Layer(output_lens[i - 1], output_lens[i], activation_fn))

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def __call__(self, x: List[Value]):
        for layer in self.layers:
            x = layer(x)
        return x