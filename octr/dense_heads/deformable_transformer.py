import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from ..ops import DeformAttn

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", num_queries=100, enc_n_points=4, dec_n_points=4):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          n_heads, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          n_heads, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers)

        self.query_embed = nn.Embedding(num_queries, d_model*2)
        
        self.reference_points = nn.Linear(d_model, 3)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, DeformAttn):
                m._reset_parameters()

        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)

    def forward(self, srcs, points, pos_embeds=None, masks=None):
        """
        srcs: (bs, n, c)
        reference_points: (bs, n, 3)
        pos_embeds: (bs, n, c)
        masks: (bs, n)
        """
        # encoder
        # 输入为srcs, (bs, n, c)
        # pos_embeds (bs, n, c)
        # masks (bs, n)
        # 输出为memory，编码后的特征表示，shape为(bs, n, c)
        memory = self.encoder(srcs, points, pos_embeds, masks)

        # prepare input for decoder
        bs, _, c = memory.shape

        # (300, 256), (300, 256)
        query_embed, tgt = torch.split(self.query_embed.weight, c, dim=1)
        # (bs, 300, 256), (bs, 300, 256)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        # (bs, 300, 3) 通过全连接层生成proposal参考点的归一化坐标(cx, cy, cz)
        reference_points = self.reference_points(query_embed).sigmoid()

        # decoder
        # output: (n_dec_layers, bs, 300, 256)
        output = self.decoder(tgt, reference_points, memory, points, query_embed, masks)

        return output


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = DeformAttn(d_model, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, src_points, pos=None, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), src_points, src, src_points, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, src_points, pos=None, padding_mask=None):
        output = src
        for _, layer in enumerate(self.layers):
            output = layer(output, src_points, pos, padding_mask)

        return output

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = DeformAttn(d_model, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, reference_points, src, src_points, query_pos, src_padding_mask=None):
        
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # src是Encoder输出的memory，即编码后的特征(bs, n_feat_points, d_model=256)，其会经过线性变换得到value
        # tgt来自于self-attention的输出，而query_pos依旧如同Decoder传进来时相同
        
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points, src, src_points, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, tgt, reference_points, src, src_points, query_pos=None, src_padding_mask=None):

        # tgt和query_pos是预设的embedding，reference points通过这个query_pos经过全连接层得到，最后一维为3
        # src是Encoder最终编码输出的特征图，即memory

        # (bs, n_query=300, hidden_dim=256)
        output = tgt
        for _, layer in enumerate(self.layers):
            output = layer(output, reference_points, src, src_points, query_pos, src_padding_mask)

        return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        n_heads=args.n_heads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        num_queries=300,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points)


