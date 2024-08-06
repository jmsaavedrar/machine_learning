#diabetes
import linear_models.linreg as linreg
import metrics.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as datasets

# reading data
X, y = datasets.load_diabetes(return_X_y=True)
# Use only one feature (bmi index = 2)
X = X[:,2:3]
print(X.shape)

# Split the data into training/testing sets
# training  80% testing 20%
n = X.shape[0]
n_valid = int(np.rint(0.2 * n)) 

# random sort
idx = np.random.permutation(n)
X = X[idx] 
y = y[idx]

X_train = X[:-n_valid]
X_valid = X[-n_valid:]
y_train = y[:-n_valid]
y_valid = y[-n_valid:]


model = linreg.LinearRegression()
coeff = model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
v = metrics.r2score(y_valid, y_pred)
print(coeff)
print('v:{}'.format(v))
#ploting
plt.scatter(X_train, y_train, color="gray")
line_x = np.array([[np.min(X)], [np.max(X)]])
line_y = model.predict(line_x)
plt.plot(line_x, line_y, marker = 'o', color = 'red' )
plt.ylabel('y (diabetes progression)')
plt.xlabel('x (body mass index)')
plt.show()

