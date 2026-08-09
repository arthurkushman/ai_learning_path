import numpy as np

mat = np.array([[1, 2, 3], [4, 5, 6]])
print(mat.shape)    # (2, 3)
print(mat.ndim)     # 2
print(mat.size)     # 6
print(mat.dtype)    # int64 (or int32 depending on platform)

arr = np.zeros((3, 4))
print(arr)
print(np.ones((2, 3)))
print(np.full((2, 3), 7.0))
print(np.eye(3))
print(np.arange(0, 10, 2) )
print(np.linspace(0, 1, 5))
print(np.random.rand(2, 3))

scores = np.array([88, 92, 75, 64, 99])
print(scores[scores > 80])

print(scores[(scores > 70) & (scores < 90)])

# 🏋️Assignment of the Day
print(np.arange(10, 35, 5) )
print(np.random.rand(3, 3))
print(np.full((2, 4), 7.5))

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a * b)
print(a @ b)
print(np.dot(a, b))
print(np.sqrt(a))
print(np.ones((2, 3)) * 5)

data = np.arange(1, 21).reshape(4, 5)
print(data[:, 3])
print(data[:2, 2:])
data[data > 10] = -1
print(data)

rand_data = np.random.randint(0, 100, 50)
print(rand_data[rand_data % 3 == 0])

print(rand_data.mean())
print(rand_data.min())
print(rand_data.max())

x = np.arange(12)
print(x)

y = x.reshape(3, 4)
print(y)
print(y.T)

image = np.arange(16).reshape(4,4)
bright_image = image + 50
print(bright_image)
print(f"original: {image}, brightened {bright_image}, clipped: {np.clip(bright_image, 0, 255)}")