import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

##Linear regression with multiple variables

path = 'data/housing_scale.txt'
X,y = load_svmlight_file(path)
# print(X,'  ',y,'\n X.shape[0]:',X.shape)

X_train, X_dev, y_train, y_dev = train_test_split(X, y,test_size=0.3, random_state = 20, shuffle=True)

theta = np.random.randn(X.shape[1])
print(X.shape[1])

def costFunction(X, y, theta):
    inner = np.power(X * theta.T - y, 2)
    m = X.shape[0]
    return np.sum(inner)/(2*m)

def gradientDescent(X, y, theta, alpha, epoch):

    cost = np.zeros(epoch)
    vcost = np.zeros(epoch)
    m = X.shape[0]

    for i in range(epoch):
        term = theta - (alpha/m) * (X*theta.T  - y).T * X
        theta = term
        # print("theta: ", theta)
        # print("theta.shape: ", theta.shape)
        cost[i] = costFunction(X, y, theta)
        vcost[i] = costFunction(X_dev,y_dev,theta)
        if i % 100 == 0:
            print("train Cost after iteration %i: %f" % (i, cost[i]))
            print("dev Cost after iteration %i: %f" % (i, vcost[i]),' \n ')

    return theta, cost, vcost

alpha = 0.5
epoch = 6000

final_theta, cost, vcost = gradientDescent(X_train, y_train, theta, alpha, epoch)

print(costFunction(X, y, theta), ' \n ', final_theta)
print(vcost)

plt.plot([i for i in range(len(cost))], cost, color="blue", label="train_loss")
plt.plot([i for i in range(len(vcost))], vcost, color="red", label="valid_loss")
plt.legend()
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("learning rate: %f" % alpha)
plt.savefig('./result.png')
plt.show()