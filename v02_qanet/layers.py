#!/usr/bin/python
# coding:utf-8

import torch
import torch.nn as nn
import math, copy
import torch.nn.functional as F
from torch.autograd import Variable

def clones(module, N):
    "produce N identical leyers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Layernorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(Layernorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(features))
        self.gamma = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True)
        return self.gamma * (x-mean)/(variance + self.eps) + self.beta


def dot_product_attention(query, key, values, mask=None, dropout=None):
    """

    :param query: [batch, h, len_q, d_k]
    :param key:   [batch, h, len_kv, d_k]
    :param values: [batch, h, len_kv, d_v]
    :param mask:
    :param dropout:
    :return:
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)  # [batch, h, len_q, len_kv]
    # print(scores.size(), mask.size())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, value=-1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = F.dropout(p_attn)
    outputs = torch.matmul(p_attn, values)
    return outputs, p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, projection=True):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.proj = projection

    def forward(self, query, key, value, mask=None):
        """

        :param query: [batch, len_q ,d_model]
        :param key:   [batch, len_kv, d_model]
        :param value: [batch, len_kv, d_model]
        :param mask:  [batch, len_kv, d_model]
        :return:
        """
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch = query.size(0)

        # line projection
        if self.proj:
            query, key, value = [linear(x).view(batch, -1, self.h, self.d_k).transpose(1,2)
                                 for linear, x in zip(self.linears, (query, key, value))]
        else:
            query, key, value = [x.view(batch, -1, self.h, self.d_k) for x in (query, key, value)]
        # Apply attention on all the projected vectors in batch
        x, self.attn = dot_product_attention(query, key, value, mask, dropout=self.dropout)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1,2).contiguous().view(batch, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PointwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ffn, dropout=0.1):
        super(PointwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)   # [max_len, 1]
        div_term = torch.exp((torch.arange(0, d_model, 2) * -(math.log(10000)/d_model)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(dim=0)    # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)    # [1, sequence_len, d_model]
        return self.dropout(x)


if __name__ == "__main__":
    d_model = 16
    heads = 8
    dropout = 0.1
    src = torch.randn(32, 10, 16)
    # src_mask = torch.sum(src,dim=-1)
    multiattention = MultiHeadAttention(heads, d_model)
    outputs = multiattention.forward(src, src, src, mask=None)
    print(outputs.size())   # [32, 10, 512]

    position = PositionalEncoding(d_model, dropout)
    outputs = position(src)
    print(outputs)



    # a_mask = torch.ones(2,2,6)
    # a = torch.Tensor([[[1,2,3,4,5,0], [2,3,4,0,0,0]], [[1,2,3,4,5,0], [2,3,4,0,0,0]]]).float()
    # b = a.masked_fill(a_mask == 0.0, value=-1e9)
    # print(b.size())