import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])

print(A @ B)
print(A * A)
print(np.linalg.det(A @ B))

M = [[2, 1, 0], [1, 3, 1], [0, 1, 2]]
print(np.linalg.inv(M))
print(M @ np.linalg.inv(M))
print(np.allclose(M, np.linalg.inv(M)))

b = [1, 2, 3]
x = np.linalg.solve(M, b)
print(np.linalg.solve(M, b))
print(M @ x)

S = [[4, 1], [1, 3]]
eigenvalues, eigenvectors = np.linalg.eigh(S)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

R = np.random.randn(4, 3)
print(R)
U, s, Vt = np.linalg.svd(R)
print("U shape:", U.shape)
print("Singular values:", s)
print("Vt shape:", Vt.shape)

# Reconstruct
S = np.zeros((4, 3))
np.fill_diagonal(S, s)
reconstructed = U @ S @ Vt
print("reconstructed shape:", reconstructed)
print(np.allclose(R, reconstructed))

A = np.array([[0, 1],
              [1, 1],
              [2, 1],
              [3, 1]])   # design matrix
y = np.array([0.5, 2.0, 3.5, 4.0])
coeff, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
print("Coefficients (m, c):", coeff)

points = np.array([[0,0], [3,4], [6,8]])
print(np.linalg.norm(points, axis=1))