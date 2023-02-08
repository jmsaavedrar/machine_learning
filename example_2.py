import csv
import linear_models.linreg as linreg
import metrics.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

filename = 'data/Housing.csv'
X = []
y = []
with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    
    for i, row in enumerate(csv_reader) :        
        if i > 0:
            y.append(float(row[0]))            
            features = [float(row[1]), float(row[2])]
            X.append(features)             
     
X = np.array(X)[:, 0:1]
y = np.array(y)


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


model = linreg.LinearRegression()
coeff = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
v = metrics.r2score(y_test, y_pred)
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


