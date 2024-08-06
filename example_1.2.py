import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import linear_models.linreg as linreg
import metrics.metrics as metrics

if __name__ == '__main__' : 
    df = pd.read_csv('data/income.data.csv')
    
    x = df['income']
    y = df['happiness'] 
    
    n = x.shape[0]
    n_valid = int(np.rint(0.2 * n)) 

    # random sort
    idx = np.random.permutation(n)
    x = x[idx] 
    x = np.expand_dims(x, axis = 1)
    y = y[idx]

    x_train = x[:-n_valid]
    y_train = y[:-n_valid]

    x_valid = x[-n_valid:]
    y_valid = y[-n_valid:]

    #ploting
    plt.scatter(x_train, y_train, color="lightblue")       
    plt.ylabel('y (happiness)')
    plt.xlabel('x (income)')
    

    #training
    model = linreg.LinearRegression()
    coeff = model.fit(x_train, y_train)    
    #prediction
    y_pred = model.predict(x_valid)    
    #metric
    v = metrics.r2score(y_valid, y_pred)
    print(coeff)
    print('v:{}'.format(v))

    line_x = np.array([[np.min(x)], [np.max(x)]])
    line_y = model.predict(line_x)
    plt.plot(line_x, line_y, marker = 'o', color = 'red' )
    plt.xlim(0,8)
    plt.ylim(0,8)
    plt.show()