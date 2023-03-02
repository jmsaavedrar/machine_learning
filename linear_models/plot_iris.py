import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE




tsne = TSNE(n_components = 2, random_state = 0)
data_2d = tsne.fit_transform(iris.data)
color={0:'red', 1: 'blue', 2:'green'}
legend = ['setosa', 'versicolor', 'virginica']
setosa = data_2d[0:50, :]
versicolor = data_2d[50:100, :]
virginica = data_2d[100:150, :]
plt.scatter(setosa[:,0], setosa[:,1], c= 'r')
plt.scatter(versicolor[:,0], versicolor[:,1], c='b')
plt.scatter(virginica[:,0], virginica[:,1], c = 'g')
plt.legend(['setoasa','versicolor','virginica'])
plt.show()
# X = iris.data
# y = iris.target
#
# print(X.shape)
# print(y.shape)




