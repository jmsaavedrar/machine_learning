import numpy as np
import nn.perceptron as perceptron
import sklearn.datasets as datasets
import metrics.metrics as metrics

#dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype = np.float64)
y = np.array([0,0,0,1], dtype = np.float64)

n = X.shape[0]
 
#Perceptron
model = perceptron.Perceptron()
model.setLoss('bce')
model.setLearningRate(1)
model.setSteps(20)
coeff = model.fit(X, y)

#Evaluation (accuracy x clase)
y_pred =model.predict(X)
acc= metrics.accuracy(np.expand_dims(y, axis = 1) , y_pred)
print('Acc Test {}'.format(acc))
#S

