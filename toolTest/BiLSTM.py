#-----------------------------------------------------
# Editor:Chanyeol Liu    Date:2020/01/05
# Code:CL200105
# Purpose:使用PyTorch搭建BiLSTM样例代码。
# https://www.jiqizhixin.com/articles/2018-10-24-13
#-----------------------------------------------------

import torch
import torch.nn as nn

class BLSTM(nn.Module):
    def __init__(self):
        return