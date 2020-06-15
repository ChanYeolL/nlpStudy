import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = np.array([[1,2,3], [1,5,4], [1,6,8]])
# theta = np.array([1,2,3])
y = np.array([1,2,3])
#
# def computeCost(X, y, theta):
#     inner = np.power(((X * theta.T) - y), 2)
#     return np.sum(inner)/(2*len(X))

# print(computeCost(X,y,theta))

# dates = pd.date_range('20130101',periods=6)
# df = pd.DataFrame(np.arange(24).reshape((6,4 )),index=dates, columns=['A','B','C','D'])
#
# print(df)
# df['E']=pd.Series([1,2,3,4,5,6],index=pd.date_range('20130101',periods=6))
# print(df)

# data = pd.Series(np.random.randn(1000),index=np.arange(1000))
# data = data.cumsum()

data = pd.DataFrame(np.random.randn(1000,4),
                    index=np.arange(1000),
                    columns=list("ABCD"))
data = data.cumsum()

ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class 1')
data.plot.scatter(x='A',y='C',color='DarkGreen',label='Class 2',ax=ax)
plt.show()

# def normalEqn(X, y):
#
#     theta = np.linalg.inv(X.T@X)@X.T@y
#
#     return theta
#
# print(normalEqn(X,y))
#
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
#
# def cost(theta, X, y):
#     theta = np.matrix(theta)
#     X = np.matrix(X)
#     y = np.matrix(y)
#     frist = np.multiply(-y, np.log(sigmoid(X* theta.T)))
#     second = np.multiply((1 - y), np.log(1 - sigmoid(X* theta.T)))
#     return np.sum(frist - second) / (len(X))