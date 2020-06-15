import numpy as np
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import scipy.sparse

path_train = 'data/a9a.txt'
X_train,y_train = load_svmlight_file(path_train)
path_test = 'data/a9a.t.txt'
X_valid, y_valid = load_svmlight_file(path_test)
#
print(X_train.shape)
print(X_valid.shape)

X_train = X_train.A.reshape(123, -1)
y_train = y_train.reshape(1, -1)
# 将标签转换成 0-1 标签
y_train = np.maximum(y_train, 0)

X_valid = X_valid.A.reshape(122, -1)
y_valid = y_valid.reshape(1, -1)
# 将标签转换成 0-1 标签
y_valid = np.maximum(y_valid, 0)

print(type(X_valid))
# 补齐缺失的一维
X_valid = np.insert(X_valid, 122, np.zeros((1, X_valid.shape[1])), axis=0)

print(X_valid.shape)

#
def initialize_with_random(dim):
    w = np.random.normal(size=(dim, 1))
    b = np.random.normal(size=(1))

    return w, b
#
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))
#
def propagate(w, b, X, Y):

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1.0 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1.0 / m) * np.dot(X, (A - Y).T)
    db = (1.0 / m) * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)  # 把shape中为1的维度去掉
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

#此函数通过运行梯度下降算法来优化w和b
def optimize(w, b,  X_train, y_train, X_valid, y_valid, num_iterations, learning_rate, print_cost=False):

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X_train, y_train)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录损失
        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

#使用学习到的w和b来预测数据集X的标签。
def predict(w, b, X_train, y_train, X_valid, y_valid, threshold=0.5):

    train_m = X_train.shape[1]
    train_Y_pred = np.zeros((1, train_m))
    train_A = sigmoid(np.dot(w.T, X_train) + b)

    for i in range(train_A.shape[1]):
        train_Y_pred[0, i] = 1 if train_A[0, i] > threshold else 0

    valid_m = X_valid.shape[1]
    valid_Y_pred = np.zeros((1, valid_m))
    valid_A = sigmoid(np.dot(w.T, X_valid) + b)

    for i in range(valid_A.shape[1]):
        valid_Y_pred[0, i] = 1 if valid_A[0, i] > threshold else 0

    train_acc = 100 - np.mean(np.abs(train_Y_pred - y_train)) * 100
    valid_acc = 100 - np.mean(np.abs(valid_Y_pred - y_valid)) * 100

    return train_acc, valid_acc

#
def model(X_train, y_train, X_valid, y_valid, num_iterations=2000, learning_rate=0.5, print_cost=False):

    # 1.初始化参数
    w, b = initialize_with_random(X_train.shape[0])

    # 2.梯度下降学的模型参数  计算损失
    parameters, grads, costs = optimize(w, b, X_train, y_train, X_valid, y_valid, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    d = {"costs": costs,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d
#
iters = 10000
d = model(X_train, y_train, X_valid, y_valid, num_iterations = iters, learning_rate = 0.5, print_cost = True)

costs = d['costs']
x = [i for i in range(0, iters, 100)]
plt.plot(x, costs, color="red", label="train_loss")
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.savefig('./result1.png')
plt.show()

th = 0.0
for i in range(0, 10):
    train_acc, valid_acc = predict(d['w'], d['b'], X_train, y_train, X_valid, y_valid, th)
    print('threshold: {}'.format( th ))
    print('train acc: {}'.format( train_acc ))
    print('valid acc: {}'.format( valid_acc ))
    print()
    th += 0.1