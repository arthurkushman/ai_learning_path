import math

class Layer:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return f"Layer: {self.name}"

    def forward(self, x):
        raise NotImplementedError("forward pass not implemented")

class DenseLayer(Layer):
    def __init__(self, name):
        super().__init__(name)

        self.in_features = None
        self.out_features = None

    def __str__(self):
        return f"DenseLayer('{self.name}', in={self.in_features}, out={self.out_features})"

    def forward(self, x):
        self.in_features = x
        self.out_features = sum(x)

        return self.out_features

class ActivationLayer(Layer):
    def __init__(self, name):
        super().__init__(name)
        self.function  = "relu"

    def __str__(self):
        return f"ActivationLayer('{self.name}', function={self.function})"

    def forward(self, x):
        if self.function == "relu":
            return [max(0, v) for v in x]
        if self.function == "sigmoid":
            return [1/(1+math.exp(-v)) for v in x]
        else:
            raise ValueError

dense = DenseLayer("dense")
activation = ActivationLayer("activation")
print(dense)
print(activation)

df = dense.forward([1, 2, 3, 4])
af = activation.forward([5, 6, 7, 8])
print(df)
print(af)