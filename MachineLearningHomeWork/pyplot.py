#-----------------------------------------------------
# Editor:Chanyeol Liu    Date:2019/11/22
# Code:CL191122
# Purpose:matplotlib.pyplot test
#-----------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
a = [1, 10000, 3, 4] # y 是 a的值，x是各个元素的索引
b = [5, 6, 7, 8]

plt.plot(a, b, 'ro--', markersize=8)
plt.xlabel('this is x')
plt.ylabel('this is y')
plt.title('this is a demo')
plt.legend() # 将样例显示出来
plt.plot()
plt.show()