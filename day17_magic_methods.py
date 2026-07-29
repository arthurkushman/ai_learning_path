class MetricTracker:
    def __init__(self):
        self.list = []
    def __len__(self):
        return len(self.list)
    def __getitem__(self, item):
        return self.list[item]
    def __repr__(self):
        return repr(self.list)
    def update(self, value):
        self.list.append(value)
    def average(self):
        return sum(self.list) / len(self.list)
    def __iter__(self):
        return iter(self.list)
    def __contains__(self, item):
        return item in self.list

mt = MetricTracker()
mt.update(0.8)
mt.update(0.5)
mt.update(0.3)

print(mt.average())

for i in mt:
    print(i)


class Normalizer:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
    def __call__(self, x):
        if self.max_val == self.min_val:
            return 0.0
        else:
            return (x - self.min_val) / (self.max_val - self.min_val)


norm = Normalizer(0, 100)
print("=== normalization tests ===")
print(norm(50))
print(norm(0))
print(norm(100))

