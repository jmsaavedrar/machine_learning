import numpy as np
import nn.perceptron as perceptron
import sklearn.datasets as datasets
import metrics.metrics as metrics
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
model = perceptron.Perceptron()
model.setLoss('mse')

print(y_train)
coeff = model.fit(X_train, y_train)

#Evaluation (accuracy x clase)
y_pred =model.predict(X_test)
acc= metrics.accuracy(np.expand_dims(y_test, axis = 1) , y_pred)
print('Acc Test {}'.format(acc))
#