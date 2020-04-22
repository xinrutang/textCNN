import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class textCNN(nn.Module):
    def __init__(self, args):
        super(textCNN, self).__init__()
        self.args = args
        Vocab = args.embed_num  ## 已知词的数量
        Dim = args.embed_dim  ##每个词向量长度
        Cla = args.class_num  ##类别数
        Ci = 1  ##输入的channel数
        Knum = args.kernel_num  ## 每种卷积核的数量
        Ks = args.kernel_sizes  ## 卷积核list，形如[2,3,4]

        self.embed = nn.Embedding(Vocab, Dim)  ## 词向量，这里直接随机
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])  ## 卷积层
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(Ks) * Knum, Cla)  ##全连接层

    def forward(self, x):
        # （8, 250）
        x = self.embed(x)
        # （8, 250, 50）
        x = x.unsqueeze(1)
        #  （8, 1, 250, 50）
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        #  [(8, 256, 249), (8, 256, 248), (8, 256, 247)] 256个卷积核；249是element-wise运算后的大小
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        # [(8, 256), (8, 256), (8, 256)]
        x = torch.cat(x, 1)  # (N,Knum*len(Ks))
        # [8, 768]
        x = self.dropout(x)
        # [8, 768]
        logit = self.fc(x)
        # [8, 5]
        return logit