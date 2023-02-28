import numpy as np

A = np.array([[1,2,3,1],[4,5,6,4],[7,8,9,7]])
B = np.array([[1],[2],[3]])
x= B * A
print(A.shape)
print(B.shape)
print(x)