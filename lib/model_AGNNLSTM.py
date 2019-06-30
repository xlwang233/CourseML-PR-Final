import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(self, requires_grad=True):
        super(GraphAttentionLayer, self).__init__()
        if requires_grad:
            # uniform initialization
            self.beta = Parameter(torch.Tensor(1).uniform_(
                0, 1), requires_grad=requires_grad)
        else:
            self.beta = Variable(torch.zeros(1), requires_grad=requires_grad)

    def forward(self, x, adj):
        # NaN grad bug fixed at pytorch 0.3. Release note:
        #     `when torch.norm returned 0.0, the gradient was NaN.
        #     We now use the subgradient at 0.0, so the gradient is 0.0.`
        norm2 = torch.norm(x, 2, 1).view(-1, 1)

        # add a minor constant (1e-7) to denominator to prevent division by
        # zero error
        cos = self.beta * \
            torch.div(torch.mm(x, x.t()), torch.mm(norm2, norm2.t()) + 1e-7)

        # neighborhood masking (inspired by this repo:
        # https://github.com/danielegrattarola/keras-gat)
        mask = (1. - adj) * -1e9
        masked = cos + mask

        # propagation matrix
        P = F.softmax(masked, dim=1)

        # attention-guided propagation
        output = torch.mm(P, x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (16 -> 16)'


class GaANLayer(nn.Module):
    def __init__(self):
        super(GaANLayer, self).__init__()

    def forward(self, key, value, adj):
        norm2 = torch.norm(x, 2, 1).view(-1, 1)

        # compute cosine similarity
        cos = self.beta * \
              torch.div(torch.mm(x, x.t()), torch.mm(norm2, norm2.t()) + 1e-7)

        # neighborhood masking (inspired by this repo:
        # https://github.com/danielegrattarola/keras-gat)
        # mask = (1. - adj) * -1e9
        # masked = cos + mask     先不搞mask这种操作，之后可以考虑搞上

        # 计算出cosine similarity，我们直接用这个similarity来做加权
        return 0


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, initializer=nn.init.xavier_uniform_):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(initializer(
            torch.Tensor(in_features, out_features)))

    def forward(self, input):
        # no bias
        return torch.mm(input, self.weight)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class AGNN_LSTM(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers, dropout_rate):
        super(AGNN_LSTM, self).__init__()

        self.layers = nlayers
        self.dropout_rate = dropout_rate

        self.embeddinglayer = LinearLayer(nfeat, nhid)
        nn.init.xavier_uniform_(self.embeddinglayer.weight)

        self.attentionlayers = nn.ModuleList()
        # for Cora dataset, the first propagation layer is non-trainable
        # and beta is fixed at 0
        self.attentionlayers.append(GraphAttentionLayer(requires_grad=True))
        for i in range(1, self.layers):
            self.attentionlayers.append(GraphAttentionLayer())

        self.outputlayer = LinearLayer(nhid, nclass)
        # nn.init.xavier_uniform(self.outputlayer.weight)  # duplicate initialization process ?

        self.lstm = nn.LSTM(input_size=156, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 156)
    def forward(self, x, speed_data, adj):
        x = F.relu(self.embeddinglayer(x))
        # x = F.dropout(x, self.dropout_rate, training=self.training)

        for i in range(self.layers):
            x = self.attentionlayers[i](x, adj)

        x = self.outputlayer(x)
        # x = F.dropout(x, self.dropout_rate, training=self.training)
        # print("coefficient size: ", x.shape)
        # print("coefficient:\n", x)
        # modify
        x = F.softmax(x, dim=1)  # row elements sum to 1
        x = torch.matmul(speed_data, x)  # (64, 3, 156)  (batch, seq, features)
        # print("speed size: ", x.shape)
        # print("speed:\n", x)
        r_out, h_n = self.lstm(x, None)  # x: (64, 1, 64)tensor containing the output features from the last layer
        # print("r_out size:", r_out.shape)
        # print("r_out last:\n", r_out[:, -1, :])
        # print("r_out last several:\n", r_out[:3, -1, :])
        out = self.fc(r_out[:, -1, :])  # (64, 1, 156)
        # return F.log_softmax(x, dim=1)
        return out
