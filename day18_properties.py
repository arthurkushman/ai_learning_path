class HyperParameters:
    dataset_size = 10000

    def __init__(self):
        self._learning_rate = 0.1
        self._batch_size = 1
        self._epochs = 1
        self._total_steps = 0

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epochs(self):
        return self._epochs

    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0:
            raise ValueError("Learning rate must be positive")
        self._learning_rate = value

    @batch_size.setter
    def batch_size(self, value):
        if value <= 0:
            raise ValueError("Batch size must be positive")
        self._batch_size = value

    @epochs.setter
    def epochs(self, value):
        if value < 1:
            raise ValueError("Epochs must be positive")
        self._epochs = value

    @property
    def total_steps(self):
        return self._epochs * (self.dataset_size // self._batch_size)

    def __str__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate}, batch_size={self.batch_size})"

hp = HyperParameters()
hp.learning_rate = 0.3
hp.batch_size = 10
hp.epochs = 1

print(str(hp))
print(hp.total_steps)