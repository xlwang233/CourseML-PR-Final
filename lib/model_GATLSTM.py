import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, originated from https://arxiv.org/abs/1710.10903
    Some of the architecture has been modified for traffic forecasting
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  # F
        self.out_features = out_features  # F'
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input_data, adj):
        # embedding = input_data[:-1]
        # speed = input_data[-1]
        h = torch.mm(input_data, self.W)  # (N, F)  (F, F') = (N, F')
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # e's size is (N, N)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # the so-called masked attention
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)  # (N, N)
        # h_prime = torch.matmul(attention, h)  # (N, F')

        if self.concat:
            h_prime = torch.matmul(attention, h)  # (N, F')
            return F.elu(h_prime)
        else:
            return attention

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionFilter(nn.Module):
    '''
    Almost the same as GraphAttentionLayer
    Only some small modifications.
    '''
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionFilter, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  # F
        self.out_features = out_features  # F'
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, embedding, x, adj):
        # embedding = input_data[:-1]
        # speed = input_data[-1]
        h = torch.mm(embedding, self.W)  # (N, F)  (F, F') = (N, F')
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # e's size is (N, N)

        #zero_vec = -9e15*torch.ones_like(e)
        #attention = torch.where(adj > 0, e, zero_vec)  # the so-called masked attention
        attention = F.softmax(e, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)  # (N, N) 暂时不要dropout了
        # h_prime = torch.matmul(attention, h)  # (N, F')

        x = torch.transpose(x, dim0=0, dim1=1)  # (156, 50)
        #x = torch.squeeze(x)
        x = torch.mm(attention, x)
        x = torch.transpose(x, dim0=0, dim1=1)  # transpose back to (50, 156) i.e. (N, M)
        return x


class GAT_LSTM_OLD(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):  # noutput is the last layer's hidden dim
        """Dense version of GAT."""
        super(GAT_LSTM_OLD, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

        self.lstm = nn.LSTM(input_size=156, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 156)
    def forward(self, feats, speed_data, adj):
        feats = F.dropout(feats, self.dropout, training=self.training)
        feats = torch.cat([att(feats, adj) for att in self.attentions], dim=1)  # (N, K*F')
        feats = F.dropout(feats, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))  # (N, noutput)
        coef = self.out_att(feats, adj)  # x: the final attention coefficient, also the weight for local speed (N, N)

        # speed_data (64, 4, 156)
        # historical = speed_data[:, :-1, :]  # (64, 3, 156)
        # predict = speed_data[:, -1, :]  # (64, 156)
        x = torch.matmul(speed_data, coef)   # (64, 3, 156)  (batch, seq, features)
        # return out_speed  # output speed rather than softmax(features), which is used for classification.
        r_out, h_n = self.lstm(x, None)  # x: (64, 1, 64)tensor containing the output features from the last layer
        out = self.fc(r_out[:, -1, :])  # (64, 1, 156)
        return out

class GAT_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):  # noutput is the last layer's hidden dim
        """Dense version of GAT."""
        super(GAT_LSTM, self).__init__()
        self.dropout = dropout

        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        #
        # self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

        self.attention = GraphAttentionFilter(nfeat, nhid, dropout, alpha)
        self.lstm = nn.LSTM(input_size=156, hidden_size=8, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=8, out_features=156)
    def forward(self, embedding, x, adj):
        # feats = F.dropout(embedding, self.dropout, training=self.training)
        # feats = torch.cat([att(feats, adj) for att in self.attentions], dim=1)  # (N, K*F')
        # feats = F.dropout(feats, self.dropout, training=self.training)
        # # x = F.elu(self.out_att(x, adj))  # (N, noutput)
        # coef = self.out_att(feats, adj)  # x: the final attention coefficient, also the weight for local speed (N, N)
        #
        # # speed_data (64, 4, 156)
        # # historical = speed_data[:, :-1, :]  # (64, 3, 156)
        # # predict = speed_data[:, -1, :]  # (64, 156)
        # x = torch.matmul(speed_data, coef)   # (64, 3, 156)  (batch, seq, features)
        # # return out_speed  # output speed rather than softmax(features), which is used for classification.
        # r_out, h_n = self.lstm(x, None)  # x: (64, 1, 64)tensor containing the output features from the last layer
        # out = self.fc(r_out[:, -1, :])  # (64, 1, 156)
        # return out

        N, T, M = x.shape
        x_attention = []
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
            # x_T = torch.unsqueeze(x_T, dim=2)  # shape (N, M, 1)
            x_T = self.attention(embedding, x_T, adj)  # return (N, M)
            x_attention.append(x_T)

        x = torch.stack(x_attention)

        # 构造lstm的输入
        # convert from (T, N, M) to (N, T, M)
        x_lstm = torch.transpose(x, dim0=0, dim1=1)

        r_out, h_n = self.lstm(x_lstm, None)  # last of the rout size: (50, 1, hiddensize)
        x = r_out[:, -1, :]

        # Softmax层
        x = self.fc(x)
        # x = tf.nn.dropout(x, dropout)
        # 返回前向计算结果
        return x