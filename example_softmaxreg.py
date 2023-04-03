from sklearn import datasets
import linear_models.softmaxreg as softmaxreg
import umap
import matplotlib.pyplot as plt
import numpy as np
import metrics.metrics as metrics

iris = datasets.load_iris()
X = iris.data
y = iris.target
 
 
# Split the data into training/testing sets
# training  80% testing 20%
n = X.shape[0]
n_test = int(np.rint(0.2 * n)) 
 
# random sort
idx = np.random.permutation(n)
X = X[idx, :] 
y = y[idx]
 
X_train = X[:-n_test, :]
y_train = y[:-n_test]
 
X_test = X[-n_test:,:]
y_test = y[-n_test:]


""" data normalization, improve convergence """
#"X_Train 150 x4 d = 4
#"mu es un vector de tamaño 4
#"dst es un vector de tamaño 4 
mu = np.mean(X_train, axis = 0)
dst = np.std(X_train, axis = 0)
X_train = (X_train - mu) / dst
X_test = (X_test - mu) / dst
"""-------------------------------------------"""
 
SM = softmaxreg.SoftmaxReg(3)
coeff = SM.fit(X_train, y_train)
""" prediction on test """
y_pred = SM.predict(X_test)
acc = metrics.multiclass_accuracy(y_test, y_pred)
print('acc {}'.format(acc))
#  
# # show confusion matrix
# print(y_pred.shape)
# print(y_test.shape)
cm = metrics.confusion_matrix(y_test, y_pred, 3)

print('confusion matrix')
print(cm)
#  
# # Just data visualization
# #  #project 2D
# # reducer = umap.UMAP()
# # print('UMAP', flush = True)
# # reducer.fit(X) 
# # embedding = reducer.transform(X)
# # print(embedding.shape)
# # plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Paired') 
# # plt.show()