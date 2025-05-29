import numpy as np
import matplotlib.pyplot as plt
x = np.array([[30, 0], 
              [30, 10],
              [10, 10],
              [20, 20],
              [20, 30],
              [10, 30],
              [10, 40]])

# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

m = x.mean(axis = 0, keepdims = True)
xn = x - m
print(xn)
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

z = np.cov(np.transpose(xn))
a, v = np.linalg.eig(z)
idx = np.argsort(-a)
v = v[:, idx]
a = a[idx]
# x = (np.array([[10,20]]) - m )
print(x)
y = xn @ v[:,0]
print('y: {}'.format(y))
for i, p in enumerate(y) : 
    xr = p  *  v[:,0]
    error = np.sqrt(np.square(xr - xn[i,:]).sum())
    print('{} -> {} error {}'.format(x[i,:], p, error))
print('xr {}'.format(xr))
print(v)
print(a)
print(idx)