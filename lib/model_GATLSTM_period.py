import torch
import torch.nn as nn
import torch.nn.functional as F

"""
加上period信息，并做成multi-head   先不做multi-head试一下。
加上validation
"""

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
    我要加masked attention看看
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

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # the so-called masked attention
        attention = F.softmax(attention, dim=1)
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

        # self.attentions = [GraphAttentionFilter(nfeat, nhid, dropout, alpha, concat=True) for _ in range(nheads)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        #
        # self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

        self.attention_recent = GraphAttentionFilter(nfeat, nhid, dropout, alpha)
        self.attention_period = GraphAttentionFilter(nfeat, nhid, dropout, alpha)
        self.lstm_recent = nn.LSTM(input_size=156, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm_period = nn.LSTM(input_size=156, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=32*2, out_features=156)
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

        # N, T, M = x.shape  # (50, 11, 156)
        recent = x[:, :4, :]  # (50, 4, 156)
        period = x[:, 4:, :]  # (50, 7, 156)
        N, T_recent, M = recent.shape
        _, T_period, _ = period.shape

        x_attention_recent = []
        x_attention_period = []

        recent = torch.transpose(recent, dim0=0, dim1=1)
        for s in range(T_recent):
            x_T = self.attention_recent(embedding, recent[s, :, :], adj)
            x_attention_recent.append(x_T)  # (T_recent, N, M)  (4, 50, 156)

        period = torch.transpose(period, dim0=0, dim1=1)
        for t in range(T_period):
            x_T = self.attention_period(embedding, period[t, :, :], adj)
            x_attention_period.append(x_T)  # (T_recent, N, M)  (7, 50, 156)

        x_recent = torch.stack(x_attention_recent)
        x_period = torch.stack(x_attention_period)

        # 构造lstm的输入
        # convert from (T, N, M) to (N, T, M)
        x_lstm_recent = torch.transpose(x_recent, dim0=0, dim1=1)  # (50, 4, 156)
        x_lstm_period = torch.transpose(x_period, dim0=0, dim1=1)  # (50, 7, 156)

        r_out_recent, h_n = self.lstm_recent(x_lstm_recent, None)  # last of the rout size: (50, 1, hiddensize)
        r_out_period, _ = self.lstm_period(x_lstm_period, None)
        x_recent = r_out_recent[:, -1, :]  # (N, hiddendim)
        x_period = r_out_period[:, -1, :]  # (N, hiddendim)

        x = torch.cat([x_recent, x_period], dim=1)  # (N, 2*hiddendim) (50, 16)

        # Softmax层
        x = self.fc(x)
        # x = tf.nn.dropout(x, dropout)
        # 返回前向计算结果
        return x
