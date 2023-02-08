#diabetes
import numpy as np
from sklearn  import datasets, linear_model
from sklearn.metrics  import r2_score


X, y = datasets.load_diabetes(return_X_y=True)

# Use only one feature (bmi index = 2)
X = X[:,2:3]
print(X.shape)

# Split the data into training/testing sets
# training  80% testing 20%
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


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
print("Intercept: \n", regr.intercept_)
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


