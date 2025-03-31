from abc import ABC, abstractmethod

class Optim(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for param in self.params:
            param.grad = 0.0

class SDG(Optim):
    def __init__(self, params, learning_rate=0.01):
        super().__init__(params)
        self.learning_rate = learning_rate
        
    def step(self):
        for param in self.params:
            param.val -= self.learning_rate * param.grad