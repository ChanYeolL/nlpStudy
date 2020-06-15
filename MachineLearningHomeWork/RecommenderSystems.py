#-----------------------------------------------------
# Editor:Chanyeol Liu    Date:2019/11/26
# Code:CL191126
# Purpose:Recommender Systems
#-----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import scipy.optimize as opt

#Movie ratings dataset

data = scio.loadmat('data/ex8_movies.mat')
print(data.keys())
R = data['R']
Y = data['Y']
nm, nu = Y.shape  # Y中0代表用户没有评分

print('Average rating for movie 1 (Toy Story): %f / 5\n\n'%np.mean(Y[0, np.where(R[0, :]==1)]))
#可视化评分矩阵
#MATLAB中imagesc(A)将矩阵A中的元素数值按大小转化为不同颜色，并在坐标轴对应位置处以这种颜色染色
plt.figure(figsize=(8,8*(1682./943.)))
plt.imshow(Y, cmap='rainbow')
plt.colorbar() #加颜色条
plt.ylabel('Movies (%d)'%nm,fontsize=20)
plt.xlabel('Users (%d)'%nu,fontsize=20)
plt.show()

#Collaborative ﬁltering learning algorithm

parameters = scio.loadmat('data/ex8_movieParams.mat')
print(parameters.keys())
X = parameters['X']
Theta = parameters['Theta']
num_users = parameters['num_users']
num_movies = parameters['num_movies']
num_features = parameters['num_features']

#减小数据集用来更快的测试代价函数的正确性
num_users = 4
num_movies = 5
num_features = 3
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]

def serialize(X, Theta):
    return np.r_[X.flatten() ,Theta.flatten()]

def deserialize(params, nm, nu, nf):
    return np.array(params)[0:nm*nf].reshape(nm, nf), np.array(params)[nm*nf:].reshape(nu, nf)

# Collaborative ﬁltering cost function
def cofiCostFunc(params, Y, R, nm, nu, nf, lmd=0):
    X, Theta = deserialize(params, nm, nu, nf)
    # print(X,'\n', Theta)
    J = np.sum(np.square(X@Theta.T - Y)[np.where(R==1)])/2
    reg = (lmd/2) * (np.square(X).sum() + np.square(Theta).sum())
    return J + reg

print(cofiCostFunc(serialize(X, Theta), Y, R, num_movies, num_users, num_features),'\n',
      cofiCostFunc(serialize(X, Theta), Y, R, num_movies, num_users, num_features, 1.5))

# Collaborative ﬁltering gradient
def cofiGradient(params, Y, R, nm, nu, nf, lmd=0):
    X,Theta = deserialize(params, nm, nu, nf)

    X_grad = ((X@Theta.T - Y)*R)@Theta + lmd*X
    Theta_grad = ((X@Theta.T - Y)*R).T@X + lmd*Theta

    return serialize(X_grad, Theta_grad)


def checkCostFunction(params, Y, myR, nm, nu, nf, lmd=0.):
    print('Numerical Gradient \t cofiGrad \t\t Difference')

    # 分析出来的梯度
    grad = cofiGradient(params, Y, myR, nm, nu, nf, lmd)

    # 用微小的e 来计算数值梯度。
    e = 0.0001
    nparams = len(params)
    e_vec = np.zeros(nparams)

    # 每次只能改变e_vec中的一个值，并在计算完数值梯度后要还原。
    for i in range(10):
        idx = np.random.randint(0, nparams)
        e_vec[idx] = e
        loss1 = cofiCostFunc(params - e_vec, Y, myR, nm, nu, nf, lmd)
        loss2 = cofiCostFunc(params + e_vec, Y, myR, nm, nu, nf, lmd)
        numgrad = (loss2 - loss1) / (2 * e)
        e_vec[idx] = 0
        diff = np.linalg.norm(numgrad - grad[idx]) / np.linalg.norm(numgrad + grad[idx])
        # print('%0.15f \t %0.15f \t %0.15f' % (numgrad, grad[idx], diff))

print("Checking gradient with lambda = 0...")
checkCostFunction(serialize(X,Theta), Y, R, num_movies, num_users, num_features)
print("\nChecking gradient with lambda = 1.5...")
checkCostFunction(serialize(X,Theta), Y, R, num_movies, num_users, num_features, 1.5)

import codecs

def loadMovieList():
    movies = []  # 包含所有电影的列表
    with codecs.open("data/movie_ids.txt",'r',encoding = "ISO-8859-1") as f:
        for line in f:
            movies.append(' '.join(line.strip().split(' ')[1:]))
            # print(movies)
    return movies

movieList = loadMovieList()

my_ratings = np.zeros((1682,1))

my_ratings[0]   = 4
my_ratings[97]  = 2
my_ratings[6]   = 3
my_ratings[11]  = 5
my_ratings[53]  = 4
my_ratings[63]  = 5
my_ratings[65]  = 3
my_ratings[68]  = 5
my_ratings[182] = 4
my_ratings[225] = 5
my_ratings[354] = 5

for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for %s\n' % (my_ratings[i], movieList[i]))

mat = scio.loadmat('data/ex8_movies.mat')
Y, R = mat['Y'], mat['R']
Y.shape, R.shape

Y = np.c_[Y, my_ratings]  # (1682, 944)
R = np.c_[R, my_ratings!=0]  # (1682, 944)
num_movies, num_users = Y.shape
num_features = 10;

def normalizeRatings(Y, R):
    Ymean = (Y.sum(axis=1) / R.sum(axis=1)).reshape(-1,1)
    Ynorm = (Y - Ymean)*R  # 这里也要注意不要归一化未评分的数据
    return Ynorm, Ymean

Ynorm, Ymean = normalizeRatings(Y, R)
Ynorm.shape, Ymean.shape
# ((1682, 944), (1682, 1))

X = np.random.random((num_movies, num_features))
print('X',X)
Theta = np.random.random((num_users, num_features))
print('Theta',Theta)
initial_parameters = serialize(X, Theta)
lmd = 10

res = opt.minimize(fun=cofiCostFunc,
                   x0=initial_parameters,
                   args=(Ynorm, R, num_movies, num_users, num_features, lmd),
                   method='TNC',
                   jac=cofiGradient,
                   options={'maxiter': 100})

ret = res.x

fit_X, fit_Theta = deserialize(ret, num_movies, num_users, num_features)

# 所有用户的剧场分数矩阵
pred_mat = fit_X @ fit_Theta.T
my_predictions = pred_mat[:,-1] + Ymean.flatten()
print('my_predictions: ',my_predictions,'my_predictions len: ',len(my_predictions))

pred_sorted_idx = np.argsort(my_predictions)[::-1] # 排序并翻转，使之从大到小排列
print('\nTop recommendations for you:\n')
for i in range(10):
    print('Predicting rating %f for movie %s\n'% (my_predictions[pred_sorted_idx[i]],movieList[pred_sorted_idx[i]]));

print("\nOriginal ratings provided:")
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print('Rated %d for movie %s\n' % (my_ratings[i], movieList[i]))