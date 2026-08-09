import numpy as np

M = np.arange(12).reshape(3,4)
row = np.array([10, 20, 30, 40])
print(row + M)

col = np.array([100, 200, 300]).reshape(-1,1)
print(col + M)
print(row + col + M)

a = np.array([1, 2, 3, 4])
b = np.array([0.5, 2.0])
print(np.add.outer(a,b))
print(np.multiply.accumulate(a))

a = np.array([1, 5, 3])
b = np.array([4, 2, 6])
print(np.maximum(a,b))

original = np.arange(24).reshape(2,3,4)
print(original.flags['C_CONTIGUOUS']) # changes the last index faster
fortran_arr = np.asfortranarray(original)
print(fortran_arr.flags['F_CONTIGUOUS']) # changes the 1st index faster

grid = np.arange(1, 26).reshape(5,5)
print(grid[[0, 0, 4, 4],[0, 4, 4, 0]])
print(grid[grid > 15])
print(np.where(grid < 15, -1, grid))

batch = np.random.randint(0, 256, size=(4, 3, 3))
min_vals = batch.min(axis=(1,2), keepdims=True)
max_vals = batch.max(axis=(1,2), keepdims=True)
batch_norm = (batch - min_vals) / (max_vals - min_vals)
print(batch_norm)