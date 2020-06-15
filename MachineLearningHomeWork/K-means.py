#-----------------------------------------------------
# Editor:Chanyeol Liu    Date:2019/11/22
# Code:CL191122
# Purpose:K-means Clustering
#-----------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import io

#Finding closest centroids
def findClosestCentroids(X, centroids):
    idx = []
    max_dist = 1000000  # 限制一下最大距离
    for i in range(len(X)):
        minus = X[i] - centroids
        dist = minus[:,0]**2 + minus[:,1]**2
     #   print("dist: ", dist)
        if dist.min() < max_dist:
            ci = np.argmin(dist)
      #      print("ci: ", ci)
            idx.append(ci)
    return np.array(idx)

mat = loadmat('data/ex7data2.mat')
X = mat['X']
init_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = findClosestCentroids(X, init_centroids)
#print(idx)

def computeCentroids(X, idx):
    centroids = []
    for i in range(len(np.unique(idx))):  # np.unique() means K
        u_k = X[idx == i].mean(axis=0)  # 求每列的平均值 !!!!!!!!
     #   print("u_k： ",u_k)
        centroids.append(u_k)

    return np.array(centroids)

computeCentroids(X, idx)

def plotData(X, centroids, idx=None):
    """
    可视化数据，并自动分开着色。
    idx: 最后一次迭代生成的idx向量，存储每个样本分配的簇中心点的值
    centroids: 包含每次中心点历史记录
    """
    colors = ['b', 'g', 'gold', 'darkorange', 'salmon', 'olivedrab',
              'maroon', 'navy', 'sienna', 'tomato', 'lightgray', 'gainsboro'
                                                                 'coral', 'aliceblue', 'dimgray', 'mintcream',
              'mintcream']

    assert len(centroids[0]) <= len(colors), 'colors not enough '

    subX = []  # 分号类的样本点
    if idx is not None:
        for i in range(centroids[0].shape[0]):
            x_i = X[idx == i]
            subX.append(x_i)
    else:
        subX = [X]  # 将X转化为一个元素的列表，每个元素为每个簇的样本集，方便下方绘图

    # 分别画出每个簇的点，并着不同的颜色
    plt.figure(figsize=(8, 5))
    for i in range(len(subX)):
        xx = subX[i]
        plt.scatter(xx[:, 0], xx[:, 1], c=colors[i], label='Cluster %d' % i)
    plt.legend()
    plt.grid(True)
    plt.xlabel('x1', fontsize=14)
    plt.ylabel('x2', fontsize=14)
    plt.title('Plot of X Points', fontsize=16)

    # 画出簇中心点的移动轨迹
    xx, yy = [], []
    for centroid in centroids:
        xx.append(centroid[:, 0])
        yy.append(centroid[:, 1])

    plt.plot(xx, yy, 'rx--', markersize=8)
    plt.show()


plotData(X, [init_centroids])


def runKmeans(X, centroids, max_iters):
    K = len(centroids)

    centroids_all = []
    centroids_all.append(centroids)
    centroid_i = centroids
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroid_i)
        centroid_i = computeCentroids(X, idx)
   #     print("centroid_i: ", centroid_i)
        centroids_all.append(centroid_i)
    #    print("centroids_all: ", centroids_all)

    return idx, centroids_all


idx, centroids_all = runKmeans(X, init_centroids, 20)
plotData(X, centroids_all, idx)


def initCentroids(X, K):
    """随机初始化"""
    m, n = X.shape
    idx = np.random.choice(m, K)
    print("idx:",idx)
    centroids = X[idx]
    print("centroids:", centroids)

    return centroids

for i in range(3):
    centroids = initCentroids(X, 3)
    idx, centroids_all = runKmeans(X, centroids, 10)
    plotData(X, centroids_all, idx)

#===========================Image compression with K-means================================#
from skimage import io

A = io.imread('data/bird_small.png')
print(A.shape)
plt.imshow(A);
A = A/255.  # Divide by 255 so that all values are in the range 0 - 1

#  Reshape the image into an (N,3) matrix where N = number of pixels.
#  Each row will contain the Red, Green and Blue pixel values
#  This gives us our dataset matrix X that we will use K-Means on.
X = A.reshape(-1, 3)
K = 16
centroids = initCentroids(X, K)
idx, centroids_all = runKmeans(X, centroids, 10)

img = np.zeros(X.shape)
centroids = centroids_all[-1]
for i in range(len(centroids)):
    img[idx == i] = centroids[i]

img = img.reshape((128, 128, 3))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(A)
axes[1].imshow(img)




