#-----------------------------------------------------
# Editor:Chanyeol Liu    Date:2019/10/28
# Code:CL191028
# Purpose:Logistic Regression
#-----------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path = 'data/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])
print(data.head())

positive = data[data.Admitted.isin(['1'])]
negative = data[data.Admitted.isin(['0'])]

#fig代表绘图窗口(Figure)；ax代表这个绘图窗口上的坐标系(axis)，一般会继续对ax进行操作。
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(positive['Exam1'], positive['Exam2'], c='b', label='Admitted')
ax.scatter(negative['Exam1'], negative['Exam2'], s=50, c='r', marker='x', label='Not Admitted')
# 设置图例显示在图的上方
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12),ncol=3)
# 设置横纵坐标名
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
theta = np.zeros(data.shape[1])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunction(X, y, theta):
    first = (-y) * np.log(sigmoid(X@theta.T))
    second = (1-y) * np.log(1 - sigmoid(X@theta.T))
    return (first-second)/len(X)

def gradient(X, y, theta):
    return (X.T @ (sigmoid(X @ theta) - y))/len(X)

result = opt.fmin_tnc(func=costFunction, x0=theta, fprime=gradient, args=(X, y))

print(result)




