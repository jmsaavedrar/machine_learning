#diabetes
import matplotlib.pyplot as plt
import numpy as np
from sklearn  import datasets, linear_model
from sklearn.metrics  import mean_squared_error, r2_score
import linear

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
n = diabetes_X.shape[0]
# random sort
idx = np.random.permutation(n)
diabetes_X = diabetes_X[idx] 
diabetes_y = diabetes_y[idx]

XX = diabetes_X.copy()
# Use only one feature
diabetes_X = diabetes_X[:,2:3]
print(diabetes_X.shape)

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
print("Intercept: \n", regr.intercept_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))
# Simple regression
ones = np.ones((diabetes_X_train.shape[0],1))
# print(ones.shape)
# print(diabetes_X_train.shape)
X = np.concatenate([ones, diabetes_X_train], axis = 1)
y = diabetes_y_train
print(X.shape)  
coeff = linear.linearRegression(X, y)
print(coeff)
# print(XX[0])
# print(y)
#print(diabetes_y_train)
# Plot outputs
plt.scatter(diabetes_X_train, diabetes_y_train, color="blue")
plt.ylabel('y (diabetes progression)')
plt.xlabel('x (body mass index)')
plt.show()

