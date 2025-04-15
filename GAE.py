

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
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
        support = torch.mm(input, self.weight)  # GraphConvolution forward。input*weight
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout_rate):
        super(InnerProductDecoder, self).__init__()

        self.dropout = dropout_rate

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = torch.sigmoid(torch.mm(z, z.t()))
        return adj


class GCN(nn.Module):
    def __init__(self, in_dim, hidden1, hidden2, hidden3, z_emb_size, dropout_rate):#初始化
        super(GCN, self).__init__()
        self.dropout = dropout_rate

        self.gc1 = GraphConvolution(in_dim, hidden1)
        self.gc2 = GraphConvolution(hidden1, hidden2)
        self.gc3 = GraphConvolution(hidden2, hidden3)
        self.gc4 = GraphConvolution(hidden3, z_emb_size)
        self.dc = InnerProductDecoder(dropout_rate)
        self.Selfatt2 = SelfAttention(z_emb_size, z_emb_size, z_emb_size)
        self.gc11 = GraphConvolution(128, 256)
        self.gc22 = GraphConvolution(256, 128)
        self.gc33 = GraphConvolution(hidden3, hidden3)
        self.gc44 = GraphConvolution(hidden3, z_emb_size)
        self.gn = GNNLayer(hidden3, z_emb_size)
        self.trans = nn.TransformerEncoderLayer(d_model=128, dim_feedforward=256, nhead=2, dropout=0.1)

        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        n_enc_1 = hidden3
        n_enc_2 = 64
        n_z = 16
        self.Gnoise = GaussianNoise(sigma=0.1, device=device)
        self.enc_1 = nn.Linear(in_dim, n_enc_1)
        self.BN_1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.BN_2 = nn.BatchNorm1d(n_enc_2)
        self.z_layer = nn.Linear(n_enc_2, n_z)
        self.gnn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_2 = GNNLayer(n_enc_2, n_z)
        self.attn1 = AttentionWide(n_enc_2, heads=8)
        self.attn2 = AttentionWide(n_z, heads=8)
        self.dec_1 = nn.Linear(n_z, n_enc_2)
        self.dec_2 = nn.Linear(n_enc_2, n_enc_1)
        self.enc_11 = nn.Linear(n_enc_1, n_enc_1)
        self.z_layer1 = nn.Linear(n_enc_1, n_enc_1)
        self.gnn_11 = GNNLayer(n_enc_1, n_enc_1)
        self.attn11 = AttentionWide(n_enc_1, heads=8)

    def gcn_encoder(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2 = F.relu(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)

        x3 = F.relu(self.gc3(x2, adj))
        x3 = F.dropout(x3, self.dropout, training=self.training)

        emb = F.relu(self.gc4(x3, adj))
        return emb

    def gcn_decoder(self, emb):
        adj_hat = self.dc(emb)
        return adj_hat

    def forward(self, x, adj):
        emb = self.gcn_encoder(x, adj)
        adj_hat = self.gcn_decoder(emb)
        return emb, adj_hat

    def gcn_encoder2(self, x, adj):

        emb = F.relu(self.gn(x, adj))

        A_pred = self.dot_product_decode(emb)

        return emb, A_pred

    def gcn_encoder3(self, x, adj):
        enc_h1 = self.BN_1(F.relu(self.enc_1(self.Gnoise(x))))
        h1 = self.gnn_1(enc_h1, adj)
        h2 = self.gnn_2(h1, adj)
        enc_h2 = self.BN_2(F.relu(self.enc_2(self.Gnoise(enc_h1))))
        enc_h2 = (self.attn1(enc_h2, h1)).squeeze(0)+enc_h2
        z = self.z_layer(self.Gnoise(enc_h2))
        z = (self.attn2(z, h2)).squeeze(0)+z
        A_pred = self.dot_product_decode(h2)
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))

        return dec_h2, A_pred

    def gcn_encoder4(self, x, adj):
        enc_h1 = self.BN_1(F.relu(self.enc_11(self.Gnoise(x))))
        h1 = self.gnn_11(enc_h1, adj)
        z = self.z_layer1(self.Gnoise(enc_h1))
        z = (self.attn11(z, h1)).squeeze(0)+z
        A_pred = self.dot_product_decode(h1)

        return z, A_pred

    def dot_product_decode(self,Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        if active:
            output = torch.tanh(output)
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v):
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn_weights = nn.Softmax(dim=-1)(attn_score)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class SelfAttention(nn.Module):
    def __init__(self, d_in, dk, dv):
        super(SelfAttention, self).__init__()

        self.WQ = nn.Linear(d_in, dk, bias=False)
        self.WK = nn.Linear(d_in, dk, bias=False)
        self.WV = nn.Linear(d_in, dv, bias=False)
        self.scaled_dot_product_attn = ScaledDotProductAttention(dk)

    def forward(self, Q, K, V):
        q_heads = self.WQ(Q)
        k_heads = self.WK(K)
        v_heads = self.WV(V)
        attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads)
        return attn, attn_weights

class AttentionWide(nn.Module):
    def __init__(self, emb, p = 0.2, heads=8, mask=False):
        super().__init__()

        self.emb = emb
        self.heads = heads
        # self.mask = mask
        self.dropout = nn.Dropout(p)
        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, y):
        b = 1
        t, e = x.size()
        h = self.heads
        # assert e == self.emb, f'Input embedding dimension {{e}} should match layer embedding dim {{self.emb}}'

        keys = self.tokeys(x).view(b, t, h, e)
        queries = self.dropout(self.toqueries(y)).view(b, t, h, e)
        values = self.tovalues(x).view(b, t, h, e)

        # dot-product attention

        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b * h, t, t)

        # if self.mask:
        #     mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        self.attention_weights = dot
        out = torch.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

# A random Gaussian noise uses in the ZINB-based denoising autoencoder.
class GaussianNoise(nn.Module):
    def __init__(self, device, sigma=1, is_relative_detach=True):
        super(GaussianNoise,self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, device = device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x