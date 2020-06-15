import numpy as np
import pandas as pd
import torch

df_train = pd.read_csv(r'fudanNLPTest\data\train.tsv',delimiter='\t')
df_test = pd.read_csv(r'fudanNLPTest\data\test.tsv',delimiter='\t')
labels = np.array(df_train['Sentiment'])

print(labels)

a = np.arange(15).reshape(3,5)
print(a.ndim)