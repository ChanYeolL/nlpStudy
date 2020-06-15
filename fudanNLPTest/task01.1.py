import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import nltk
import pandas as pd
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

# 导入数据
df_train = pd.read_csv(r'fudanNLPTest\data\train.tsv', sep='\t')
df_test = pd.read_csv(r'fudanNLPTest\data\test.tsv', sep='\t', header=0, index_col=0)
labels = np.array(df_train['Sentiment'])
print('labels',labels)
# 数据处理
def normalize_data(words):
    def remove_none_ascii(words):
        new_words = []
        for word in words:
            """在Python中规范化（规范化）unicode数据以删除变音符号，重音等。"""
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(words):
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    words = remove_none_ascii(words)
    words = to_lowercase(words)
    return words

#torch.cuda.set_device(0)

# 一：标记短语
#word_list=list(set(df_train['Phrase']))
# #CountVectorizer： 只考虑词汇在文本中出现的频率
Vectorizer = CountVectorizer(stop_words='english', min_df=5, max_df=0.90)
#抽取特征向量
train_CountVectorizer = Vectorizer.fit_transform(df_train['Phrase'])
#转换为one-hot,用的是线性分类，矩阵化数据
train_one_hot = train_CountVectorizer.toarray()
train_bag = Vectorizer.vocabulary_
print("len(train_bag)  ",len(train_bag), train_bag)
print("train_one_hot  " , train_one_hot)

# 二：划分数据集：训练集/验证集
#一般训练集：验证集：测试集=7：2：1
split_idx0 = int(len(df_train) * 0.2)
#devtest
split_idx1 = int(len(df_train) * 0.3)
train_x, test_x = train_one_hot[:split_idx0], train_one_hot[split_idx0:split_idx1]
train_y, test_y = labels[:split_idx0], labels[split_idx0:split_idx1]
print("len(train_x)  " ,len(train_x))
print("len(train_y)  " ,len(train_y))
# 创建Tensor数据集
# 构建数据，（数据，标签）
train_data = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
test_data = TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))
print("train_data  ",train_data)
print("test_data  " ,test_data)
# 每批处理的样本的个数
batch_size = 64
#DataLoader本质上就是一个iterable（跟python的内置类型list等一样），并利用多进程来加速batch data的处理，使用yield来使用有限的内存
#DataLoader是一个高效，简洁，直观的网络输入数据结构，便于使用和扩展
# shuffle：打乱数据之间的顺序，让数据随机化
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
print('len(train_loader)',len(train_loader))
print('len(test_loader)',len(test_loader))

# 三：构建模型
#继承自nn.Module
class linear_classfy(torch.nn.Module):

    def __init__(self):
        super(linear_classfy, self).__init__()
        #Linear = ... # type: 'QSGTexture.Filtering'
        #self.logistic = torch.nn.Linear(in_dim, out_class) // in_dim = 28*28  out_class = 10
        self.linear0 = torch.nn.Linear(len(train_bag), 256)
        self.linear1 = torch.nn.Linear(256, 128)
        self.linear2 = torch.nn.Linear(128, 5)

    def forward(self, x):
        #return self.logistic(x)
        x = self.linear0(x)
        x = self.linear1(x)
        out = self.linear2(x)
        #交叉熵损失函数自带sofmax
        return out

#词汇大小，数字标签
model = linear_classfy()
#把模型移到GPU
print(torch.cuda.is_available())
#model.cuda()
print("model" , model)
# 交叉熵损失
loss_function = torch.nn.CrossEntropyLoss()
# 随机梯度下降，学习率 0.2  //另外一个优化方法 Adam
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

# 训练（需要改进）
for epoch in range(1):
    for inputs, labels in train_loader:
        x = inputs
        target = labels
        out = model(x)
        #target label不是one-hotted
        loss = loss_function(out, target)
        #把梯度置零，也就是把loss关于weight的导数变成0.如果不清零，那么使用的这个grad就得同上一个mini-batch有关
        # 一种梯度下降法
        optimizer.zero_grad()
        #反向传播，计算梯度，optimizer更新参数空间需要基于反向梯度
        loss.backward()
        #应用渐变
        #optimizer.step()通常用在每个mini-batch之中,只有用了optimizer.step()，模型才会更新
        optimizer.step()

#测试
pred_y = torch.LongTensor()
for inputs, labels in test_loader:
    x = inputs
    target = labels
    test_output = model(x)
    #torch.max():返回一个tensor中的最大值
    #Tensor.view():把tensor 进行reshape的操作.第一个参数1将第一个维度的大小设定成1，后一个-1就是说第二个维度的大小=元素总数目/第一个维度的大小.
    pred_y_onebatch = torch.max(test_output, 1)[1].view(1, -1)
    pred_y = torch.cat((pred_y, pred_y_onebatch), 1)

#计算准确度
print(pred_y,test_y)
pred_y = pred_y.cpu().numpy()
accuracy = float((pred_y == test_y).astype(int).sum()) / np.size(pred_y)
print('准确性: ', accuracy)