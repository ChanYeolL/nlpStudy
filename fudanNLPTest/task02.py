#-----------------------------------------------------
# Editor:Chanyeol Liu    Date:2019/12/15
# Code:CL191215
# Purpose:任务二：基于深度学习的文本分类
#-----------------------------------------------------
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

df_train = pd.read_csv(r'data\train.tsv',delimiter='\t')
df_test = pd.read_csv(r'data\test.tsv',delimiter='\t')
labels = np.array(df_train['Sentiment'])

print(labels)

a = np.arange(15).reshape(3,5)
print(a.ndim)