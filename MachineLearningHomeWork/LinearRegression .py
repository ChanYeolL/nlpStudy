#-----------------------------------------------------
# Editor:Chanyeol Liu    Date:2019/10/27
# Code:CL191028
# Purpose:Linear regression with one variable,
#         Linear regression with multiple variables
#-----------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Linear regression with one variable

path = 'data/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())
data.plot(kind='scatter', x='Population', y='Profit', figsize=(8,5))
plt.show()

data.insert(0, 'Ones', 1)
print(data.head())
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]
print(X.head())
print(y.head())
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0,0])
print(np.array([[0,0]]).shape, X.shape, theta.shape, y.shape)

def costFunction(X, y, theta):
    inner = np.power(X * theta.T - y, 2)
    return np.sum(inner)/(2*len(X))

def gradientDescent(X, y, theta, alpha, epoch):

    term = np.matrix(np.zeros(theta.shape))
    cost = np.zeros(epoch)
    m = X.shape[0]

    for i in range(epoch):
        term = theta - (alpha/m) * (X*theta.T  - y).T * X
        theta = term
        cost[i] = costFunction(X, y, theta)

    return theta, cost

alpha = 0.01
epoch = 1500

final_theta, cost = gradientDescent(X, y , theta, alpha, epoch)
print(costFunction(X, y, final_theta))
print(final_theta,'  ',cost)
testData = np.matrix([1,7.0032])
print('xxx', testData*final_theta.T)
##Linear regression with multiple variables

path = 'data/ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data2.head())
#Feature Normalization
data2 = (data2 - data2.mean())/data2.std()

data2.insert(0,'Ones', 1)

cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols-1]
y2 = data2.iloc[:, cols-1:cols]

X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

final_theta2, cost2 = gradientDescent(X2, y2, theta2, alpha, epoch)

print(costFunction(X2, y2, theta2), '  ', final_theta2)

def normalEqu(X, y ):
    theta = np.linalg.inv(X.T@X)@X.T@y
    return theta

final_theta3=normalEqu(X, y)#感觉和批量梯度下降的theta的值有点差距
print(final_theta3)


