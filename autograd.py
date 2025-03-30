from typing import List
from abc import ABC, abstractmethod
import math

#Only for scalar functions, i.e. from R^n to R
class DifferentiableFn(ABC):
    @abstractmethod
    def gradient(self, x: "List[Value]") -> List[float]: #The gradient of a R^n -> R function is a R^n -> R^n function
        pass
    @abstractmethod
    def __call__(self, x: "List[Value]") -> "Value":
        pass

class Value:
    def __init__(self, val, label = "", children: "List[Value]" = [], operation: DifferentiableFn | None = None, requires_grad = True):
        self.val = val
        self.children = children
        self.operation = operation
        self.label = label
        self.grad = 0.0
        self.requires_grad = requires_grad

    def __repr__(self):
        if self.label=="":
            return f"Value({self.val})"
        else:
            return f"{self.label}({self.val})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        return Add()([self, other])

    def __radd__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        return Add()([other, self])

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        return Mul()([self, other])

    def __rmul__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        return Mul()([other, self])

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        return Sub()([self, other])

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        return Sub()([other, self])

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        return Div()([self, other])

    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        return Div()([other, self])

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other, requires_grad=False)
        return Pow()([self, other])
    
    def __neg__(self):
        return Mul()([self, Value(-1.0, requires_grad=False)])

    """
    # recursive version of backpropagation
    def backward(self, start=True):
        if self.operation == None or not self.requires_grad:
            return

        if start:
            self.grad = 1.0

        grads = self.operation.gradient(self.children)
        for i, child in enumerate(self.children):
            if child.requires_grad:
                child.grad += grads[i] * self.grad
        
        for child in self.children:
            if child.requires_grad:
                child.back(start=False)
    """

    def backward(self):
        if self.operation == None or not self.requires_grad:
            return

        self.grad = 1.0

        #backpropagate trhough the computation tree with dfs
        stack = [self]
        while len(stack) != 0: 
            curr = stack.pop()
            if curr.operation == None:
                continue

            grads = curr.operation.gradient(curr.children)
            for i, child in enumerate(curr.children):
                if child.requires_grad:
                    #accumulate the gradients on each child: the grad can come from multiple paths
                    child.grad += grads[i] * self.grad 

            for child in curr.children:
                if child.requires_grad:
                    stack.append(child) 
            

    def print_children_tree(self, depth=0, indent=" |", show_grad=False):
        total_indent = indent*depth
        grad_str = "grad=" + f"{self.grad:.3f}" + ", " if show_grad and self.requires_grad else ""
        operation_str = self.operation if self.operation != None else ""
        print(f"{total_indent}{self} {grad_str}{operation_str}")

        for child in self.children:
            child.print_children_tree(depth=depth+1, indent=indent, show_grad=show_grad)

class Add(DifferentiableFn):
    def __call__(self, x: List[Value]) -> Value:
        return Value(x[0].val + x[1].val, operation=self, children=x)
    def gradient(self, _: List[Value]) -> List[float]:
        return [1.0, 1.0]
    def __repr__(self):
        return "add"

class Sub(DifferentiableFn):
    def __call__(self, x: List[Value]) -> Value:
        return Value(x[0].val - x[1].val, operation=self, children=x)
    def gradient(self, _: List[Value]) -> List[float]:
        return [1.0, -1.0]
    def __repr__(self):
        return "sub"

class Mul(DifferentiableFn):
    def __call__(self, x: List[Value]) -> Value:
        return Value(x[0].val * x[1].val, operation=self, children=x)
    def gradient(self, x: List[Value]) -> List[float]:
        return [x[1].val, x[0].val]
    def __repr__(self):
        return "mul"

class Div(DifferentiableFn):
    def __call__(self, x: List[Value]) -> Value:
        return Value(x[0].val / x[1].val, operation=self, children=x)
    def gradient(self, x: List[Value]) -> List[float]:
        return [1.0, -x[1].val**(-2.0)]
    def __repr__(self):
        return "div"

class Pow(DifferentiableFn):
    def __call__(self, x: List[Value]) -> Value:
        return Value(x[0].val ** x[1].val, operation=self, children=x)
    def gradient(self, x: List[Value]) -> List[float]:
        return [x[1].val * (x[0].val ** (x[1].val - 1)), math.log(x[0].val) * x[0].val ** x[1].val]
    def __repr__(self):
        return "pow"

class Exp(DifferentiableFn):
    def __call__(self, value: Value) -> Value:
        return Value(math.exp(value.val), operation=self, children=[value])
    def gradient(self, x: List[Value]) -> List[float]:
        return [math.exp(x[0].val)]
    def __repr__(self):
        return "exp"

class Relu(DifferentiableFn):
    def __call__(self, value: Value) -> Value:
        return Value(max(0, value.val), operation=self, children=[value])
    def gradient(self, x: List[Value]) -> List[float]:
        return [0.0] if x[0].val <= 0.0 else [1.0]
    def __repr__(self):
        return "relu"

class Tanh(DifferentiableFn):
    def __call__(self, value: Value) -> Value:
        return Value(math.tanh(value.val), operation=self, children=[value])
    def gradient(self, x: List[Value]) -> List[float]:
        return [1 - math.tanh(x[0].val)**2]
    def __repr__(self):
        return "tanh"

class Average(DifferentiableFn):
    def __call__(self, values: List[Value]) -> Value:
        avg = 0
        for value in values:
            avg += value.val
        avg /= len(values)
        return Value(avg, operation=self, children=values)

    def gradient(self, x: List[Value]) -> List[float]:
        return [1.0 / len(x) for _ in x] 

    def __repr__(self):
        return "average"