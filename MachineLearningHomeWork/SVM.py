#-----------------------------------------------------
# Editor:Chanyeol Liu    Date:2019/11/03
# Code:CL191103
# Purpose:Support Vector Machines
#-----------------------------------------------------
import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn import svm
import re
import matplotlib.pyplot as plt

import nltk, nltk.stem.porter

data1 = scio.loadmat('data/ex6data1.mat')
X1 = data1['X']
y1 = data1['y'].flatten()

def plot_data(X, y):
    p = X[y==1]
    n = X[y==0]
    print(p)
    plt.scatter(p[:,0], p[:,1], c='k', marker='+', label='y=1')
    plt.scatter(n[:,0], n[:,1], c='y', marker='o', edgecolors='k', linewidths=0.5, label='y=0')

clf1 = svm.SVC(C=1, kernel='linear')
clf1.fit(X1, y1)

def plot_boundary(clf, X1):
    u = np.linspace(np.min(X1[:,0]), np.max(X1[:,0]), 500)
    v = np.linspace(np.min(X1[:,1]), np.max(X1[:,1]), 500)
    # 转为网格（500*500）
    x, y = np.meshgrid(u, v)
    # 因为predict中是要输入一个二维的数据，因此需要展开
    z = clf.predict(np.c_[x.flatten(), y.flatten()])
    z = z.reshape(x.shape)
    # 画等高线
    plt.contour(x, y, z, 1, colors='b')
    plt.title('The Decision Boundary')

plt.figure(1)
plot_data(X1, y1)
plot_boundary(clf1, X1)
plt.show()

data2 = scio.loadmat('data/ex6data2.mat')
X2 = data2['X']
y2 = data2['y'].flatten()



def gaussianKernel(X1, X2, sigma):
    return np.exp( -((X1 - X2).T@(X1 - X2)) / (2*sigma*sigma))

a1 = np.array([1, 2, 1])
a2 = np.array([0, 4, -1])
sigma = 2
print(gaussianKernel(a1, a2, sigma))

clf2 = svm.SVC(C=1, kernel='rbf', gamma=np.power(0.1, -2)/2)
clf2.fit(X2, y2)

#画图
plt.figure(2)
plot_data(X2, y2)
plot_boundary(clf2, X2)
plt.show()


data3 = scio.loadmat('data/ex6data3.mat')
X3 = data3['X']
y3 = data3['y'].flatten()
Xval = data3['Xval']
yval = data3['yval'].flatten()

plot_data(X3, y3)
plt.show()
#交叉验证集Xval，yval
plot_data(Xval, yval)
plt.show()

try_value = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

def error_rate(predict_y, yval):
    m = yval.size
    count = 0
    for i in range(m):
        count = count + np.abs(int(predict_y[i])-int(yval[i]))
    return float(count/m)

#模型选择
def model_selection(try_value, X3, y3, Xval, yval):
    error = 1
    c = 1
    sigma = 0.01
    for i in range(len(try_value)):
        for j in range(len(try_value)):
            clf = svm.SVC(C=try_value[i], kernel='rbf', gamma=np.power(try_value[j], -2)/2)
            clf.fit(X3, y3)
            predict_y = clf.predict(Xval)
            if error > error_rate(predict_y, yval):
                error = error_rate(predict_y, yval)
                c = try_value[i]
                sigma = try_value[j]
    return c, sigma, error

c, sigma, error = model_selection(try_value, X3, y3, Xval, yval)

clf3 = svm.SVC(C=c, kernel='rbf', gamma=np.power(sigma, -2)/2)
clf3.fit(X3, y3)

#画图
plt.figure(3)
plot_data(X3, y3)
plot_boundary(clf3, X3)
plt.show()

with open('data/emailSample1.txt', 'r') as f:
    email = f.read()
    print(email)

def processEmail(email):
    """替换"""
    email = email.lower()
    email = re.sub('<[^<>]>', ' ', email)
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    email = re.sub('[\$]+', 'dollar', email)
    email = re.sub('[\d]+', 'number', email)
    return email

def email2TokenList(email):
    stemmer = nltk.stem.porter.PorterStemmer()
    email = processEmail(email)
    print("xxxx",email)
    tokens = re.split('[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)
    # 遍历每个分割出来的内容
    tokenlist = []
    for token in tokens:
        # 删除任何非字母数字的字符
        token = re.sub('[^a-zA-Z0-9]', '', token);
        # Use the Porter stemmer to 提取词根
        try:  # porterStemmer有时会出现问题,因此用try
            stemmed = stemmer.stem(token)
        except:
            stemmed = ''
        # 去除空字符串‘’，里面不含任何字符
        if not len(token): continue
        tokenlist.append(stemmed)
    return tokenlist

#得到单词表，序号为索引号+1
vocab_list = np.loadtxt('data/vocab.txt', dtype='str', usecols=1)

def email2VocabIndices(email):
    """提取存在单词的索引"""
    token = email2TokenList(email)
    print("token",token)
    index = [i+1 for i in range(len(vocab_list)) if vocab_list[i] in token ]
    print("index",index,vocab_list)
    return index

#查看样例序号
mail_indices = email2VocabIndices(email)
for i in mail_indices:
    print(i, end=' ')

def emailFeatures(vocab_indices):
    """
    将email转化为词向量。存在单词的相应位置的值置为1，其余为0
    """
    vector = np.zeros(len(vocab_list))  # init vector
    # 将有单词的索引置为1
    for i in vocab_indices:
        vector[i-1] = 1
    return vector

vector = emailFeatures(mail_indices)
print("length of vector = ",len(vector), "\n num of non-zero = ",vector.sum())
print("vector",vector)

# Training set
mat1 = scio.loadmat('data/spamTrain.mat')
X4, y4 = mat1['X'], mat1['y']

#使用email2FeatureVector函数处理每个原始电子邮件并将其转换为向量$x^{(i)}∈R^{1899}$。

print("mat1",mat1)
clf4 = svm.SVC(C=0.1, kernel='linear')
clf4.fit(X4, y4)

# Test set
mat2 = scio.loadmat('data/spamTest.mat')
Xtest, ytest = mat2['Xtest'], mat2['ytest']

predTrain = clf4.score(X4, y4)
predTest = clf4.score(Xtest, ytest)
print(predTrain, predTest)

indices = np.argsort(clf4.coef_).flatten()[::-1]  # 对权重序号进行从大到小排序 并返回
print(indices)

for i in range(15):  # 打印权重最大的前15个词 及其对应的权重
    print('{} ({:0.6f})'.format(vocab_list[indices[i]], clf4.coef_.flatten()[indices[i]]))

#打印权重最高的前15个词,邮件中出现这些词更容易是垃圾邮件
i = (clf4.coef_).size-1
while i >1883:
    #返回从小到大排序的索引，然后再打印
    print(vocab_list[np.argsort(clf4.coef_).flatten()[i]], end=' ')
    i = i-1

#垃圾邮件(y = 1)还是非垃圾邮件(y = 0)
with open('data/spamSample1.txt', 'r') as f:
    email1 = f.read()
    # print(email1)

email1Vector = np.reshape(emailFeatures(email2VocabIndices(email1)), (1,1899))
print("email1Vector",email1Vector)
email1Result = clf4.predict(email1Vector)
print(email1Result)