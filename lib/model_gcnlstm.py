import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
# from torch.nn.modules.module import Module


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvFilter(nn.Module):

    def __init__(self, Fin, Fout, bias=True):
        super(GraphConvFilter, self).__init__()
        self.Fout = Fout
        self.weight = Parameter(torch.FloatTensor(Fin, Fout))

        if bias:
            self.bias = Parameter(torch.FloatTensor(Fout))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # Fout 为卷积个数
        # N 即 batch_size的大小， M 为 x 的特征维度（即节点个数），Fin = 1 代表的一个timestep
        N, M, Fin = x.shape
        N, M, Fin = int(N), int(M), int(Fin)

        # print("N, M, Fin: ", N, M, Fin)

        x = torch.transpose(x, dim0=0, dim1=1)  # (156, 50, 1)
        # print("x size: ", x.shape)
        x = torch.squeeze(x)
        x = torch.mm(adj, x)
        x = torch.transpose(x, dim0=0, dim1=1)  # (50, 156, 1)
        x = torch.unsqueeze(x, dim=2)  # (50, 156, 1)
        x = torch.reshape(x, (N*M, Fin))  # (50*156, 1)
        x = torch.matmul(x, self.weight)  #  (50*156, 8)
        x = torch.reshape(x, (N, M, self.Fout))

        if self.bias is not None:
            return x + self.bias
        else:
            return x


class GCN_LSTM_WRONG(nn.Module):

    def __init__(self, nfeat_period, nfeat_recent, nhid, noutput, dropout):
        super(GCN_LSTM_WRONG, self).__init__()

        self.gc1 = GraphConvolution(nfeat_period, nhid)
        self.gc2 = GraphConvolution(nhid, noutput)
        self.gc3 = GraphConvolution(nfeat_recent, nhid)
        self.gc4 = GraphConvolution(nhid, noutput)
        self.dropout = dropout

        self.lstm1 = nn.LSTM(input_size=156, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=156, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128*2, 156)

    def gcfilter(self, x, adj, Fout):
        # Fout 为卷积个数
        # N 即 batch_size的大小， M 为 x 的特征维度（即节点个数），Fin = 1 代表的一个timestep
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)

        # Filter: Fin*Fout filters , i.e. one filterbank per feature pair.
        # W = self._weight_variable([Fin, Fout], regularization=False)
        weight = Parameter(torch.FloatTensor(Fin, Fout))

    def forward(self, x_period, x_recent, adj):
        # the input is (N, in_features, num_nodes), which need to be reshaped to (N, num_nodes, in_features)
        # num_nodes = x.shape[-1]
        # in_features = x.shape[1]

        x_period = F.relu(self.gc1(x_period, adj))
        x_period = F.dropout(x_period, self.dropout, training=self.training)
        x_period = self.gc2(x_period, adj)  # at this time, the size of x is (50, 156, 7)

        # x need to be reshaped to be fed into LSTM.  convert to (50, 7, 156)
        x_period = torch.transpose(x_period, dim0=1, dim1=2)

        # FOR RECENT TREND. SAME AS THE ABOVE
        x_recent = F.relu(self.gc3(x_recent, adj))
        x_recent = F.dropout(x_recent, self.dropout, training=self.training)
        x_recent = self.gc4(x_recent, adj)  # at this time, the size of x is (50, 156, 3)

        # x need to be reshaped to be fed into LSTM.  convert to (50, 7, 156)
        x_recent = torch.transpose(x_recent, dim0=1, dim1=2)

        r_out1, h_n1 = self.lstm1(x_period, None)  # last of the rout size: (50, 1, 64)
        # out1 = self.fc(r_out1[:, -1, :])  # out: (50, 1, 156)

        r_out2, h_n2 = self.lstm2(x_period, None)
        # out2 = self.fc(r_out2[:, -1, :])  # out: (50, 1, 156)

        # Concatenate the two lstm results
        out = torch.cat((r_out1[:, -1, :], r_out2[:, -1, :]), dim=1)  # out size (50, 128)
        out = self.fc(out)  # out size (50, 156)
        return out


class GCN_LSTM(nn.Module):

    def __init__(self, noutput, dropout):
        super(GCN_LSTM, self).__init__()

        self.noutput = noutput
        self.filter = GraphConvFilter(Fin=1, Fout=8)
        self.lstm = nn.LSTM(input_size=1248, hidden_size=8, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=8, out_features=noutput)
        self.dropout = dropout

    def forward(self, x, adj):
        N, T, M = x.shape

        x_gcn = []
        for s in range(int(T)):
            # 临时设置的一个用于拼接的 tensor
            a = torch.zeros([1, int(M)])
            for j in range(int(N)):
                #  timesteps 同一位置的 tensor
                x_temp = x[j, s, :]
                x_temp = torch.reshape(x_temp, [1, int(M)])
                a = torch.cat([a, x_temp], 0)
            # timesteps 同一位置的 tensor 拼接结果
            # x_T.shape = (N,M)
            x_T = a[1:, :]  # (50, 156)
            x_T = torch.unsqueeze(x_T, dim=2)  # shape (N, M, 1)
            x_T = self.filter(x_T, adj)  # return (N, M, Fout)
            N, M, F = x_T.shape
            x_T = torch.reshape(x_T, [int(N), int(M * F)])
            x_gcn.append(x_T)

        x = torch.stack(x_gcn)

        # x = torch.FloatTensor(x_gcn)
        # x 变成一个list 包含 timesteps 个 tensor， 每个tensor 为 N 个 样本的 GCN 计算结果
        # 构造lstm的输入
        # convert from (T, N, M*F) to (N, T, M*F)
        x_lstm = torch.transpose(x, dim0=0, dim1=1)

        r_out, h_n = self.lstm(x_lstm, None)  # last of the rout size: (50, 1, hiddensize)
        x = r_out[:, -1, :]

        # Softmax层
        x = self.fc(x)
        # x = tf.nn.dropout(x, dropout)
        # 返回前向计算结果
        return x


