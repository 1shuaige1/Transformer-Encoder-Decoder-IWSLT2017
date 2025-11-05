import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Attention 机制模块
# ---------------------------

def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    if dropout:
        attn = dropout(attn)
    return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        bsz = query.size(0)
        def shape(x, linear):
            return linear(x).view(bsz, -1, self.h, self.d_k).transpose(1, 2)
        q, k, v = shape(query, self.linear_q), shape(key, self.linear_k), shape(value, self.linear_v)
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(bsz, -1, self.h * self.d_k)
        return self.linear_out(x)


# ---------------------------
# Feed Forward 模块
# ---------------------------

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ---------------------------
# Positional Encoding
# ---------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, learned=False):
        super().__init__()
        if learned:
            self.pe = nn.Embedding(max_len, d_model)
            self.learned = True
        else:
            self.learned = False
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        if self.learned:
            pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            return x + self.pe(pos)
        else:
            return x + self.pe[:, :x.size(1)]


# ---------------------------
# Encoder / Decoder 结构
# ---------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, ff, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.ff = ff
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, ff, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ff = ff
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, memory_mask):
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, tgt_mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.src_attn(x2, memory, memory, memory_mask))
        x2 = self.norm3(x)
        x = x + self.dropout(self.ff(x2))
        return x


# ---------------------------
# 完整 Transformer 模型
# ---------------------------

class TransformerModel(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_layers, d_ff, dropout, max_len, pad_id, learned_pos=False):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, MultiHeadAttention(n_heads, d_model, dropout),
                         PositionwiseFeedForward(d_model, d_ff, dropout), dropout)
            for _ in range(n_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, MultiHeadAttention(n_heads, d_model, dropout),
                         MultiHeadAttention(n_heads, d_model, dropout),
                         PositionwiseFeedForward(d_model, d_ff, dropout), dropout)
            for _ in range(n_layers)
        ])
        self.src_embed = nn.Embedding(vocab, d_model)
        self.tgt_embed = nn.Embedding(vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, learned_pos)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab)

    def make_tgt_mask(self, tgt):
        batch, seq = tgt.size()
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        subsequent = torch.tril(torch.ones((seq, seq), device=tgt.device)).bool()
        return pad_mask & subsequent.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt):
        src_mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)
        tgt_mask = self.make_tgt_mask(tgt)
        x_enc = self.pos_enc(self.src_embed(src))
        for layer in self.encoder:
            x_enc = layer(x_enc, src_mask)
        x_dec = self.pos_enc(self.tgt_embed(tgt))
        for layer in self.decoder:
            x_dec = layer(x_dec, x_enc, tgt_mask, src_mask)
        return self.fc_out(self.norm(x_dec))
