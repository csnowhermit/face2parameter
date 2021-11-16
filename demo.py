import numpy as np

a = np.ndarray([1, 1])
b = np.ndarray([1, 1])

c = np.append(a, b, axis=1)
print(c.shape)