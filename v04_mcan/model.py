# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class MwAN(nn.Module):
    def __init__(self, embedding, vocab_size, embedding_size, encoder_size, drop_out=0.1):
        super(MwAN, self).__init__()
        self.drop_out=drop_out
        # embedding layer
        # self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)

        # highway network
        self.qWT = nn.Linear(embedding_size, embedding_size, bias=False)
        self.qWH = nn.Linear(embedding_size, embedding_size, bias=False)
        self.pWT = nn.Linear(embedding_size, embedding_size, bias=False)
        self.pWH = nn.Linear(embedding_size, embedding_size, bias=False)

        # encoder layer
        self.q_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.p_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.a_encoder = nn.GRU(input_size=embedding_size, hidden_size=int(embedding_size / 2), batch_first=True,
                                bidirectional=True)

        # interaction layer
        # Concat Attention
        self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, 1, bias=False)
        # Bilinear Attention
        self.Wb = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=False)
        # Dot Attention :
        self.Wd = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vd = nn.Linear(encoder_size, 1, bias=False)
        # Minus Attention :
        self.Wm = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vm = nn.Linear(encoder_size, 1, bias=False)
        # dot attention between query
        self.Ws = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vs = nn.Linear(encoder_size, 1, bias=False)
        # qanet
        self.Wqa1 = nn.Linear(2 * encoder_size, 1, bias=False)
        self.Wqa2 = nn.Linear(2 * encoder_size, 1, bias=False)
        self.Wqa3 = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=False)

        # multi-cast
        # compression function
        self.p_fc = nn.Linear(encoder_size * 4, 1, bias=False)
        self.p_fm = nn.Linear(encoder_size * 2, 1, bias=False)
        self.p_fs = nn.Linear(encoder_size * 2, 1, bias=False)
        self.q_fc = nn.Linear(encoder_size * 4, 1, bias=False)
        self.q_fm = nn.Linear(encoder_size * 2, 1, bias=False)
        self.q_fs = nn.Linear(encoder_size * 2, 1, bias=False)

        # modeling layer
        self.gru_agg = nn.GRU(16 * encoder_size + 24, encoder_size, batch_first=True, bidirectional=True)
        # add highway
        # self.aggWH = nn.Linear(2 * encoder_size, 2 * encoder_size)
        # self.aggWT = nn.Linear(2 * encoder_size, 2 * encoder_size)

        """
        prediction layer
        """
        self.a_attention = nn.Linear(embedding_size, 1, bias=False)
        self.Wq = nn.Linear(2 * encoder_size + 24, encoder_size, bias=False)
        self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.Wp1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp2 = nn.Linear(2 * encoder_size + 24, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.prediction = nn.Linear(2 * encoder_size, embedding_size, bias=False)
        self.initiation()

    def initiation(self):
        # initrange = 0.1
        # nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, inputs):
        [query, passage, answer, is_train] = inputs
        q_embedding = self.embedding(query)
        p_embedding = self.embedding(passage)
        a_embeddings = self.embedding(answer)

        # highway network
        qo_embedding = F.leaky_relu(self.qWH(q_embedding)) * F.sigmoid(self.qWT(q_embedding))
        qf_embedding = (1-F.sigmoid(self.qWT(q_embedding))) * q_embedding
        q_embedding = qo_embedding + qf_embedding

        po_embedding = F.leaky_relu(self.pWH(p_embedding)) * F.sigmoid(self.pWT(p_embedding))
        pf_embedding = (1 - F.sigmoid(self.pWT(p_embedding))) * p_embedding
        p_embedding = po_embedding + pf_embedding

        # answer self-attention
        a_embedding, _ = self.a_encoder(a_embeddings.view(-1, a_embeddings.size(2), a_embeddings.size(3)))
        a_score = F.softmax(self.a_attention(a_embedding), 1)
        a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze()
        a_embedding = a_output.view(a_embeddings.size(0), 3, -1)

        # encoder layer
        hp, _ = self.p_encoder(p_embedding)
        hp=F.dropout(hp,self.drop_out)
        hq, _ = self.q_encoder(q_embedding)
        hq=F.dropout(hq,self.drop_out)

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

        mc_sim = hq.bmm(hp.transpose(2,1))   # [batch, q_len, p_len]
        slqa_p = F.softmax(mc_sim, dim=2)  # [batch, q_len, p_len]
        slqa_p = slqa_p.bmm(hp)  # [batch, q_len, encoder_size*2] # P~
        fusion_q = F.tanh(self.Wa_o(torch.cat([hq, slqa_p, hq - slqa_p, hq * slqa_p], 2)))
        gate_q = F.sigmoid(self.Wa_g(torch.cat([hq, slqa_p, hq - slqa_p, hq * slqa_p], 2)))
        slqa_out = fusion_q * gate_q + (1 - gate_q) * slqa_p

        # multi-cast
        # max-pooling attention
        q_max = torch.max(mc_sim, dim=2, keepdim=True)[0]  # [batch, q_len, 1]
        q_max = q_max * hq
        # mean-pooling atention
        q_mean = torch.mean(mc_sim, dim=2, keepdim=True)  # [batch, q_len, 1]
        q_mean = q_mean * hq  # [batch, q_len, embed_size]
        # alignment attention
        q_align = F.softmax(mc_sim, dim=2).bmm(hp)  # [batch, q_len, embed_size]
        # intra-attention
        mc_sim_q = hq.bmm(hq.transpose(2, 1))  # [batch, q_len, q_len]
        q_intra = F.softmax(mc_sim_q, dim=2).bmm(hq)  # [batch, q_len, embed_size]

        # cast attention query
        # max pooling and sum
        q_max_fc = torch.sum(torch.cat([q_max, hq], 2), dim=2, keepdim=True)  # [batch, q_len, 1]
        q_max_fm = torch.sum(q_max * hq, dim=2, keepdim=True)  # [batch, q_len, 1]
        q_max_fs = torch.sum(q_max - hq, dim=2, keepdim=True)  # [batch, q_len, 1]
        # mean pooling and sum
        q_mean_fc = torch.sum(torch.cat([q_mean, hq], 2), dim=2, keepdim=True)  # [batch, q_len, 1]
        q_mean_fm = torch.sum(q_mean * hq, dim=2, keepdim=True)  # [batch, q_len, 1]
        q_mean_fs = torch.sum(q_mean - hq, dim=2, keepdim=True)  # [batch, q_len, 1]
        # alignment and sum
        q_align_fc = torch.sum(torch.cat([q_align, hq], 2), dim=2, keepdim=True)  # [batch, q_len, 1]
        q_align_fm = torch.sum(q_align * hq, dim=2, keepdim=True)  # [batch, q_len, 1]
        q_align_fs = torch.sum(q_align - hq, dim=2, keepdim=True)  # [batch, q_len, 1]
        # intra and sum
        q_intra_fc = torch.sum(torch.cat([q_intra, hq], 2), dim=2, keepdim=True)  # [batch, q_len, 1]
        q_intra_fm = torch.sum(q_intra * hq, dim=2, keepdim=True)  # [batch, q_len, 1]
        q_intra_fs = torch.sum(q_align - hq, dim=2, keepdim=True)  # [batch, q_len, 1]

        # max pooling and nn
        q_max_fc2 = self.q_fc(torch.cat([q_max, hq], 2))  # [batch, q_len, 1]
        q_max_fm2 = self.q_fm(q_max * hq)
        q_max_fs2 = self.q_fs(q_max - hq)
        # mean_pooling and nn
        q_mean_fc2 = self.q_fc(torch.cat([q_mean, hq], 2))  # [batch, q_len, 1]
        q_mean_fm2 = self.q_fm(q_mean * hq)
        q_mean_fs2 = self.q_fs(q_mean - hq)
        # alignment and nn
        q_align_fc2 = self.q_fc(torch.cat([q_align, hq], 2))  # [batch, q_len, 1]
        q_align_fm2 = self.q_fm(q_align * hq)
        q_align_fs2 = self.q_fs(q_align - hq)
        # intra and nn
        q_intra_fc2 = self.q_fc(torch.cat([q_intra, hq], 2))  # [batch, q_len, 1]
        q_intra_fm2 = self.q_fm(q_intra * hq)
        q_intra_fs2 = self.q_fs(q_intra - hq)

        # concat multi-cast
        hq_mc = torch.cat([ hq,
                         q_max_fc, q_max_fm, q_max_fs,
                         q_mean_fc, q_mean_fm, q_mean_fs,
                         q_align_fc, q_align_fm, q_align_fs,
                         q_intra_fc, q_intra_fm, q_intra_fs,
                         q_max_fc2, q_max_fm2, q_max_fs2,
                         q_mean_fc2, q_mean_fm2, q_mean_fs2,
                         q_align_fc2, q_align_fm2, q_align_fs2,
                         q_intra_fc2, q_intra_fm2, q_intra_fs2
                         ], 2)  # [batch, q_len, embed_size + 12]

        # aggregation
        aggregation = torch.cat([hq_mc, qts, qtc, qtd, qtb, qtm, slqa_out], 2)
        agg_rep, _ = self.gru_agg(aggregation)
        # agg_o_rep = F.leaky_relu(self.aggWH(agg_rep))
        # agg_f_rep = F.sigmoid(self.aggWT(agg_rep))
        # agg_rep = agg_o_rep * agg_f_rep + (1-agg_f_rep) * agg_rep
        aggregation_representation = F.dropout(agg_rep, self.drop_out)

        # predition layer
        sj = self.vq(torch.tanh(self.Wq(hq_mc))).transpose(2, 1) # [batch, 1, q_len]
        rq = F.softmax(sj, 2).bmm(hq_mc) # [batch, 1, 2 * encoder_size + 24]
        sj = F.softmax(self.vp(self.Wp1(aggregation_representation) + self.Wp2(rq)).transpose(2, 1), 2)
        rp = sj.bmm(aggregation_representation)
        encoder_output = F.dropout(F.leaky_relu(self.prediction(rp)),self.drop_out)
        score = F.softmax(a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze(), 1)
        if not is_train:
            return score.argmax(1)
        loss = -torch.log(score[:, 0]).mean()
        return loss


