import numpy as np
import linear_models.logreg as logreg
import sklearn.datasets as datasets
import metrics.metrics as metrics
#dataset
iris = datasets.load_iris()

x = np.array(iris.data)
y = np.array(iris.target)
c1 = 1
c2 = 2

idxs = np.where( (y == c1) | (y == c2) )[0]
x = x[idxs,:]
y = y[idxs]
y[y==c1] = 0
y[y==c2] = 1
n = x.shape[0]
n_test = int(np.rint(0.2 * n)) 

# random sort
idx = np.random.permutation(n)
X = x[idx] 
y = y[idx]

X_train = X[:-n_test]
y_train = y[:-n_test] 
X_test = X[-n_test:]
y_test = y[-n_test:]

""" data normalization, improve convergence """
mu = np.mean(X_train, axis = 0)
dst = np.std(X_train, axis = 0)
X_train = (X_train - mu) / dst
X_test = (X_test - mu) / dst
"""-------------------------------------------"""
print(y_test.shape)
print(X_test.shape)

#Logistic Regression
model = logreg.LogRegression()

print(y_train)
coeff = model.fit(X_train, y_train)

#Evaluation (accuracy x clase)
y_pred =model.predict(X_test)
acc= metrics.accuracy(np.expand_dims(y_test, axis = 1) , y_pred)
print('Acc Test {}'.format(acc))
#