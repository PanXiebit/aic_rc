#!/usr/bin/python
# coding:utf-8

import torch.nn as nn
import copy
import torch
from torch.autograd import Variable
import numpy as np
from layers import Layernorm, clones, MultiHeadAttention, PointwiseFeedForward, PositionalEncoding

# Encoder: The encoder is composed of a stack of $N=6$ identical layers.


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        """
        :param layer: a connection layer with two layers
        :param N:  in the paper, N is 6
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)  # each layer have a encodersublayer with two-sublayers, there are 6 layers
        self.norm = Layernorm(layer.size)

    def forward(self, x, mask):
        "pass the input (and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm.forward(x)


class EncoderSublayer(nn.Module):
    "the encoder sublayer with two layers, self attention and feed forward layer."
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderSublayer, self).__init__()
        self.self_attn = self_attn
        self.feed_ward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_ward)
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = Layernorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """ note that normalization first before instead of last. and dropout last.
        :param x: [batch, sequence_len, num_units]
        :param sublayer: two layers, self attention and feed forward networks
        :return:
        """
        output = self.dropout(self.norm.forward(sublayer(x)))
        return output + x


class Conv_block(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer: a connection layer with two layers
        :param N:  in the paper, N is 6
        """
        super(Conv_block, self).__init__()
        self.layers = clones(layer, N)  # each layer have a encodersublayer with two-sublayers, there are 6 layers
        self.norm = Layernorm(layer.size)

    def forward(self, x):
        "pass the input (and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x)
        return self.norm.forward(x)

class Convsublayer(nn.Module):
    def __init__(self, size, sep_conv, dropout):
        super(Convsublayer, self).__init__()
        self.size = size
        self.sep_conv = sep_conv
        self.sublayer = SublayerConnection(size, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.sublayer(x, self.sep_conv)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(
                in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(
                in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(
                in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(
                in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        x = x.transpose(2,1)
        x = self.pointwise_conv(self.depthwise_conv(x))
        return x.transpose(2,1)



if __name__ == "__main__":
    heads = 8
    embed_size = 32
    d_model = 32
    dropout = 0.1
    N = 1
    c = copy.deepcopy
    # multi_attn = MultiHeadAttention(heads, d_model)
    # ffn = PointwiseFeedForward(d_model, d_ffn=d_model*4)
    # pe = PositionalEncoding(d_model, dropout, max_len=500)
    # encodersublayer = EncoderSublayer(d_model, c(multi_attn), c(ffn), dropout)
    # encoder= Encoder(encodersublayer, N)
    # print(encoder)
    #
    src = Variable(torch.randn(2, 10, embed_size))
    # src_mask = Variable(torch.Tensor([[[2],[1]], [[2],[0]]]))
    # print(src.size())
    # print(src_mask.size())
    #
    # outputs = encoder.forward(src, src_mask)
    # print(outputs)

    sep_conv = DepthwiseSeparableConv(embed_size, d_model, k=5)
    conv_block = Conv_block(Convsublayer(d_model, sep_conv, dropout), 1)
    print(conv_block.forward(src))
