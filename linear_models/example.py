
import matplotlib.pyplot as plt
import numpy as np
from  sklearn import linear_model

def expand_data(x, order) :
    xp = np.ones(x.shape)
    for i in range(order) :
        xp = np.vstack((xp, np.power(x, i + 1)))
    xp = np.transpose(xp)
    return xp
    
#build dataset
n_train = 10
n_test  = 100
x_train = np.random.rand(n_train)
x_train = np.sort(x_train)

x_test = np.random.rand(n_test)
x_test = np.sort(x_test)

y = np.sin(2*np.pi*x_train)
y_test = np.sin(2*np.pi*x_test)

# plt.plot(x_train, y, 'b')
# plt.scatter(x_test, y_test, s=50, color = 'g')
# plt.show()

noise = np.random.normal(loc = 0, scale = 0.1, size = n_train)
y_noise = y + noise
#linear regresion
#data preparation
order = 3
xp_train = expand_data(x_train, order = order)
xp_test = expand_data(x_test, order = order)

model = linear_model.LinearRegression()
#model = linear_model.ElasticNet(alpha = 0)
model.fit(xp_train, y_noise)
yt = model.predict(xp_train)
yt_test = model.predict(xp_test)
# plots
fig, ax = plt.subplots(1,2)
ax[0].set_ylim([-2,2])
ax[0].set_title('Training (order {})'.format(order))
ax[1].set_ylim([-2,2])
ax[1].set_title('Testing  (order {})'.format(order))
ax[0].plot(x_train, y, 'b')
ax[0].plot(x_train, yt, 'r')
ax[1].plot(x_test, y_test, 'b')
ax[1].plot(x_test, yt_test, 'r')
plt.show()


