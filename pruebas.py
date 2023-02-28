import numpy as np

# A = np.array([[1,2,3,1],[4,5,6,4],[7,8,9,7]])
# B = np.array([[1],[2],[3]])
# x= B * A
# print(A.shape)
# print(B.shape)
# print(x)
A = np.array([[1,2,3,1],[4,5,6,4],[7,8,9,7]])
X = np.array([[0,0,0,0],[1,1,1,1]])
y = np.array([0,1])

A[y,:] = A[y,:]  +  X
s = np.sum(A, axis = 1, keepdims = True)
print(y==1)
print(A)
print(A[np.where(y==1)[0],:])

