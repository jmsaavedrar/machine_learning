import numpy as np
import linear_models.logreg as logreg
import sklearn.datasets as datasets
#dataset
iris = datasets.load_iris()
X = iris.data[:100,:]
y = iris.target[:100]

n = X.shape[0]
n_test = int(np.rint(0.2 * n)) 

# random sort
idx = np.random.permutation(n)
X = X[idx] 
y = y[idx]

X_train = X[:-n_test] 
X_test = X[-n_test:]
y_train = y[:-n_test]
y_test = y[-n_test:]

print(y_test.shape)
print(X_test.shape)
#Logistic Regression
model = logreg.LogRegression()
# std = np.std(X_train, axis = 0)
# mu = np.mean(X_train, axis = 0)
# X_train = (X_train - mu ) / std
print(y_train)
coeff = model.fit(X_train, y_train)


#Evaluation (accuracy x clase)

#