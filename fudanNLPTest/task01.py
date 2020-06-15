import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv(r'data\train.tsv', sep='\t')
df_test = pd.read_csv(r'data\test.tsv', sep='\t', header=0, index_col=0)
df_train.head()
# labels = np.array(df_train['Sentiment'])

pltx = np.array(df_train['PhraseId'])
plty = np.array(df_train['Sentiment'])
print(min(plty),'  ',max(plty))
plt.title("data demo")
plt.xlabel("Phrase")
plt.ylabel("Sentiment")
plt.plot(pltx,plty,"ob")
plt.show()

orig_data = df_train.values
print(orig_data)
cols = orig_data.shape[1]
X = orig_data[:,0:cols-2]
print("666  ", X)
y = orig_data[:,cols-1:cols]


class LinearRegression:
    def costFunction(self, X, y, theta):
        Inner = np.power(X*theta.T - y , 2)
        return np.sum(Inner)/(2*len(X))




class LogisticRegression:

    def __init__(self):
        #初始化Linear Regression模型
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def shuffleData(self, data):
        np.random.shuffle(data)
        cols = orig_data.shape[1]
        X = orig_data[:, 0:cols - 1]
        y = orig_data[:, cols - 1:]
        return X, y

    # h(x)
    def model(self, X, theta):
        return self._sigmoid(self, X.dot(theta))

    def fit(self, X_train, y_train, alpha=0.01, n_iters=1e4):

        #正规化的代价函数
        def costReg(theta, X, y , learningRate):
            theta = np.matrix(theta)
            X = np.matrix(X)
            y = np.matrix(y)
            first = np.multiply(-y, np.log(self.model(X, theta)))
            second = np.multiply((1 - y), np.log(1 - self.model(X, theta)))
            reg = (learningRate / (2 * len(X))* np.sum(np.power(theta[:,1:theta.shape[1]], 2)))
            return np.sum(first - second) / (len(X)) + reg

        def gradientCost(theta, X, y):
            return  np.dot(X.T, self.model(self, X, theta) - y) / len(X)

        def gradient_descent(X, y, initial_theta, alpha, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = gradientCost(theta, X, y)
                last_theta = theta
                theta = theta - alpha * gradient
                if (abs(costReg(theta, X, y) - costReg(last_theta, X, y)) < epsilon):
                    break

                cur_iter += 1

            return theta
        # print("2222  ", X_train, theta)
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])

        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, alpha, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict_proba(self, X_predict):
        #给定待预测数据集X_predict，返回表示X_predict的结果概率向量

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self.sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        #给定待预测数据集X_predict，返回表示X_predict的结果向量

        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        #根据测试数据集 X_test 和 y_test 确定当前模型的准确度

        y_predict = self.predict(X_test)
        accuracy = float((y_predict == y_test).astype(int).sum()) / np.size(y_predict)
        return accuracy

    def __repr__(self):
        return "LogisticRegression()"

## print("444  ", X, theta)
print(LogisticRegression.fit(LogisticRegression,X,y))

class SoftmaxRegression:

    def cost(self, err, label_data):
        '''
        :param err: exp的值
        :param label_data: 标签的值
        :return: 损失函数的值
        '''
        m = np.shape(err)[0]
        sum_cost = 0.0
        for i in range(m):
            if err[i, label_data[i, 0]] / np.sum(err[i, :]) > 0:
                sum_cost -= np.log(err[i, label_data[i, 0]] / np.sum(err[i, :]))
            else:
                sum_cost -= 0
        return sum_cost / m

    def gradientAscent(self, feature_data, label_data, k, maxCycle, alpha):
        '''利用梯度下降法训练Softmax模型
        :param feature_data: 特征
        :param label_data: 标签
        :param k: 类别个数
        :param maxCycle: 最大迭代次数
        :param alpha: 学习率
        :return weights: 权重
        '''
        m, n = np.shape(feature_data)
        weights = np.mat(np.ones((n, k)))  # 初始化权重
        i = 0
        while i <= maxCycle:
            err = np.exp(feature_data * weights)
            if i % 100 == 0:
                print("\t--------iter:", i, ",cost:", self.cost(err, label_data))
                rowsum = -err.sum(axis=1)
                rowsum = rowsum.repeat(k, axis=1)
                err = err / rowsum
                for x in range(m):
                    err[x, label_data[x, 0]] += 1
                weights = weights + (alpha / m) * feature_data.T * err
                i += 1
            return weights

SoftmaxRegression