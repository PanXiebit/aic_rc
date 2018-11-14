# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
from encoder import Encoder, EncoderSublayer, DepthwiseSeparableConv, Conv_block, Convsublayer
from layers import MultiHeadAttention, PointwiseFeedForward, PositionalEncoding
import copy


class MwAN(nn.Module):
    def __init__(self, embedding, vocab_size, heads, embedding_size,
                 encoder_size, conv_num=4, attn_num=1, drop_out=0.2):
        super(MwAN, self).__init__()
        self.drop_out=drop_out
        self.conv_num = conv_num
        self.attn_num = attn_num
        self.c = copy.deepcopy

        # self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)
        self.position_encoding = PositionalEncoding(embedding_size, self.drop_out, max_len=500)

        # projection
        self.proj_a = nn.Linear(embedding_size, encoder_size)
        self.proj_p = nn.Linear(embedding_size, encoder_size)
        self.proj_q = nn.Linear(embedding_size, encoder_size)

        # depthwise separable convolution
        self.sep_conv = DepthwiseSeparableConv(encoder_size, encoder_size, k=5, dim=1)
        self.conv_block = Conv_block(Convsublayer(
            encoder_size, self.c(self.sep_conv), self.drop_out), N=2)
        # encoder using self-attention
        self.attn = MultiHeadAttention(heads, encoder_size, self.drop_out, projection=False)
        self.ffn = PointwiseFeedForward(encoder_size, encoder_size * 4)
        self.encoder = Encoder(EncoderSublayer(
            encoder_size, self.c(self.attn), self.c(self.ffn), self.drop_out), N=3)
        self.a_proj = nn.Linear(encoder_size, embedding_size)

        self.a_attention = nn.Linear(embedding_size, 1, bias=False)
        # Concat Attention
        self.Wc1 = nn.Linear(encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, 1, bias=False)
        # Bilinear Attention
        self.Wb = nn.Linear(encoder_size, encoder_size, bias=False)
        # Dot Attention :
        self.Wd = nn.Linear(encoder_size, encoder_size, bias=False)
        self.vd = nn.Linear(encoder_size, 1, bias=False)
        # Minus Attention :
        self.Wm = nn.Linear(encoder_size, encoder_size, bias=False)
        self.vm = nn.Linear(encoder_size, 1, bias=False)
        # dot attention between query
        self.Ws = nn.Linear(encoder_size, encoder_size, bias=False)
        self.vs = nn.Linear(encoder_size, 1, bias=False)

        # qanet
        self.Wqa1 = nn.Linear(encoder_size, 1, bias=False)
        self.Wqa2 = nn.Linear(encoder_size, 1, bias=False)
        self.Wqa3 = nn.Linear(encoder_size, encoder_size, bias=False)

        # modeling layer
        # add highway
        self.aggWH = nn.Linear(8 * encoder_size, 8 * encoder_size)
        self.aggWT = nn.Linear(8 * encoder_size, 8 * encoder_size)
        self.agg_linear = nn.Linear(8 * encoder_size, encoder_size * 2)
        self.agg_sep_conv = DepthwiseSeparableConv(encoder_size * 8, encoder_size * 8, k=5, dim=1)
        self.agg_conv_block = Conv_block(Convsublayer(
            encoder_size * 8, self.c(self.agg_sep_conv), self.drop_out), N=2)
        # encoder using self-attention
        self.agg_attn = MultiHeadAttention(heads, encoder_size * 8, self.drop_out, projection=False)
        self.agg_ffn = PointwiseFeedForward(encoder_size * 8, encoder_size * 4)
        self.agg_encoder = Encoder(EncoderSublayer(
            encoder_size * 8, self.c(self.agg_attn), self.c(self.agg_ffn), self.drop_out), N=4)
        """
        prediction layer
        """
        self.Wq = nn.Linear(encoder_size, encoder_size, bias=False)
        self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.Wp1 = nn.Linear(encoder_size * 8, encoder_size, bias=False)
        self.Wp2 = nn.Linear(encoder_size, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.prediction = nn.Linear(encoder_size * 8, embedding_size, bias=False)
        self.initiation()

    def initiation(self):
        # initrange = 0.1
        # nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, inputs):
        [query, passage, answer, is_train] = inputs
        # embed layer
        q_embedding = self.embedding(query)
        p_embedding = self.embedding(passage)
        a_embeddings = self.embedding(answer)
        a_embedding = a_embeddings.view(-1, a_embeddings.size(2), a_embeddings.size(3))

        # position encoding
        q_embedding = self.position_encoding(q_embedding)
        p_embedding = self.position_encoding(p_embedding)
        a_embedding = self.position_encoding(a_embedding)

        # projection
        q_proj = self.proj_q(q_embedding)
        p_proj = self.proj_p(p_embedding)
        a_proj = self.proj_a(a_embedding)

        # separable conv and self-attention
        a_mask = (answer.view(-1, answer.size(2)) != 0).unsqueeze(-2)
        a_embedding = self.conv_block(a_proj)
        a_embedding = self.encoder(a_embedding, mask=a_mask) # [batch*3, max_answer, encoder_size]
        a_embedding = self.a_proj(a_embedding)
        a_score = F.softmax(self.a_attention(a_embedding), 1) # [batch*3, max_answer, 1]
        a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze() # [batch*3, embed_size]
        a_embedding = a_output.view(a_embeddings.size(0), 3, -1) # [batch, 3, embed_size]

        p_mask = (passage != 0).unsqueeze(-2)
        p_conv = self.conv_block(p_proj)
        hp = self.encoder(p_conv, mask=p_mask)
        hp=F.dropout(hp,self.drop_out)

        q_mask = (query != 0).unsqueeze(-2)
        q_conv = self.conv_block(q_proj)
        hq = self.encoder(q_conv, mask=q_mask)
        hq = F.dropout(hq,self.drop_out)

        # concat
        _s1 = self.Wc1(hp).unsqueeze(1)   # [batch, 1, passage_len, encoder_size]
        _s2 = self.Wc2(hq).unsqueeze(2)   # [batch, query_len, 1, encoder_size]
        sjt = self.vc(torch.tanh(_s1 + _s2)).squeeze()   # [batch, query_len, passage_len]
        ait = F.softmax(sjt, 2)
        qtc = ait.bmm(hp)  # [batch, query_len, encoder_size]

        # bilinear
        _s1 = self.Wb(hp).transpose(2, 1)  # [batch, encoder_size, passage_len]
        sjt = hq.bmm(_s1)   # [batch, query_len, passage_len]
        ait = F.softmax(sjt, 2)
        qtb = ait.bmm(hp)

        # dot
        _s1 = hp.unsqueeze(1)
        _s2 = hq.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtd = ait.bmm(hp)

        # minus
        sjt = self.vm(torch.tanh(self.Wm(_s1 - _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qtm = ait.bmm(hp)

        # dot, self-attention of query
        _s1 = hq.unsqueeze(1)           # [batch, 1, query_len, encoder_size]
        _s2 = hq.unsqueeze(2)           # [batch, query_len, 1, encoder_size]
        sjt = self.vs(torch.tanh(self.Ws(_s1 * _s2))).squeeze()
        ait = F.softmax(sjt, 2)
        qts = ait.bmm(hq)

        # qanet
        q_len = query.shape[1]
        p_len = passage.shape[1]
        subres0 = self.Wqa1(hp).repeat(1,1,q_len).transpose(2,1)   # [batch, query_len, passage_len]
        subres1 = self.Wqa2(hq).repeat(1,1,p_len)                  # [batch, query_len, passage_len]
        subres3 = self.Wqa3(hq).bmm(hp.transpose(2,1))             # [batch, query_len, passage_len]
        sim_mat = subres0 + subres1 + subres3                      # [batch, query_len, passage_len]]
        S_ = F.softmax(sim_mat, 2)       # [batch, query_len, passage_len]]
        p2q = S_.bmm(hp)                 # [batch, query_len, encoder_size]
        S_T = F.softmax(sim_mat, 1).transpose(2,1)  # [batch, passage_len, query_len]
        q2p = S_.bmm(S_T).bmm(hq)

        # aggregation
        aggregation = torch.cat([hq, qts, qtc, qtd, qtb, qtm, p2q, q2p], 2)
        # aggregation = torch.cat([hq, p2q, q2p], 2)
        # agg_o_rep = F.leaky_relu(self.aggWH(aggregation))
        # agg_f_rep = F.sigmoid(self.aggWT(aggregation))
        # agg_rep = agg_o_rep * agg_f_rep + (1-agg_f_rep) * aggregation
        aggregation = F.dropout(aggregation, self.drop_out)
        agg_conv = self.agg_conv_block(aggregation)
        aggregation_representation = self.agg_encoder(agg_conv, mask=q_mask)
        # print(aggregation_representation.size())

        # predition layer
        sj = self.vq(torch.tanh(self.Wq(hq))).transpose(2, 1)
        rq = F.softmax(sj, 2).bmm(hq)
        aaa = self.Wp1(aggregation_representation)
        bbb = self.Wp2(rq)
        sj = F.softmax(self.vp(aaa + bbb).transpose(2, 1), 2)
        rp = sj.bmm(aggregation_representation)
        encoder_output = F.dropout(F.leaky_relu(self.prediction(rp)),self.drop_out)
        score = F.softmax(a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze(), 1)
        if not is_train:
            return score.argmax(1)
        loss = -torch.log(score[:, 0]).mean()
        return loss


