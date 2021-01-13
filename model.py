"""
ref:
  - Attention is all you need 
  - [実装](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy

class EncoderDecoder(nn.Module):
    """
    メインモデル  
    embedded data -> encoder -> decoder -> embededd data

    args:
      - d_model (int) : ベクトルの次元
      - N_enc (int) : encoder layerの数
      - N_dec (int) : decoder layerの数
      - h_enc (int) : エンコーダのマルチヘッドの分割数
      - h_dec (int) : デコーダのマルチヘッドの分割数  
    """
    def __init__(self, d_model=512, N_enc=6, N_dec=6, h_enc=8, h_dec=8):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(d_model, N_enc, h_enc)
        self.decoder = Decoder(d_model, N_dec, h_dec)
    
    def forward(self, x_emb, y_emb, mask):
        """
        args:
          - x_emb (torch.tensor) (B x len x d) : 入力データ(位置エンコーディングまで終わったもの)
          - y_emb (torch.tensor) (B x len x d) : 出力データ(位置エンコーディングまで終わったもの)
        """
        z = self.encode(x_emb)
        y_out = self.decode(y_emb, z, mask)
        return y_out
    
    def encode(self, x_emb):
        z = self.encoder(x_emb)
        return z
    
    def decode(self, y_emb, z, mask):
        y = self.decoder(y_emb, z, mask)
        return y

class Encoder(nn.Module):
    """
    エンコーダ全体
    embedded data -> encoder_layer -> x N -> hidden data(to decoder)

    args:
      - d_model (int) : ベクトルの次元数
      - N (int) : encoder_layerの層数
      - h (int) : マルチヘッドの分割数
    """
    def __init__(self, d_model, N, h):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(N):
            self.layers.append(EncoderLayer(d_model, h))
    
    def forward(self, x_emb):
        """
        args:
          - x_emb (torch.tensor) (B x len x d) : 位置エンコーディングまで終わったもの
        """
        for layer in self.layers:
            x_emb = layer(x_emb)
        return x_emb

class EncoderLayer(nn.Module):
    """
    エンコーダのそれぞれのブロック
    input -> (residual path) -> Multihead attention -> add&norm -> (residual path) -> feed forward -> add&norm -> out

    args:
      - d_model (int) : ベクトルの次元数
      - h (int) : マルチヘッドの分割数
    """
    def __init__(self, d_model, h=8, dropout=0):
        super().__init__()
        self.multi_attention = MultiHeadAttention(h,d_model,dropout)
        self.layer_norm_1 = LayerNorm(d_model)
        self.feedforward = FeedForward(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
    
    def forward(self, x):
        res = x
        out = self.multi_attention(x,x,x)
        out = self.layer_norm_1(res + out)
        res = out
        out = self.feedforward(out)
        out = self.layer_norm_2(res + out)
        return out


class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details).
    refのサイトよりコピペ
    args;
      - features (int) : 各レイヤのベクトルの次元数
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class FeedForward(nn.Module):
    """
    単語ごとの全結合層

    - args:
      - d (int) : 次元数
      - hid_dim (int) : 隠れ層の次元数(2048デフォルト)
      - dropout (float) : dropout ratio
    """
    def __init__(self, d, hid_dim=2048, dropout=0.0):
        super().__init__()
        self.l1 = nn.Linear(d, hid_dim)
        self.l2 = nn.Linear(hid_dim, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l2(self.dropout(F.relu(self.l1(x))))

def attention(query, key, value, mask=None, dropout=0.0):
    """
    Scale Dot-Product Attention (論文Fig.2)

    args:
      - query (torch.tensor) (B x len x d)
      - key (torch.tensor) (B x len x d)
      - value (torch.tensor) (B x len x d)
      - mask (torch.tensor) (len x len) : デコーダの時に使うマスク (マスクするべきところは0, それ以外は1)
      - dropout (float) : ドロップアウトの割合(使用するなら)
    
    return:
      - out (torch.tensor) (B x len x d)
    """
    d_k = key.shape[-1]
    scale = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)

    if mask is not None:# decoing
        scale = scale.masked_fill(mask == 0, -1e9)# -infで埋めるイメージ。めちゃめちゃ確率小さくなる
    atn = F.softmax(scale, dim=-1)
    if dropout is not None:# ここにはさむべき？？
        atn = F.dropout(atn, p=dropout)   
    out = torch.matmul(atn, value)
    return out


class MultiHeadAttention(nn.Module):
    """
    エンコーダデコーダフラグが必要。 -> maskで十分

    qkv -> linear -> multi_qkv -> attention each head -> concat -> linear
    each of qkv (B x len x d) -------> linear (B x len x d)

    args:
      - h (int) : number of multi head
      - d (int) : number of dimention of hidden vector
      - dropout (float) : dropout ratio
    """
    def __init__(self, h, d, dropout=0):
        super().__init__()
        assert d%h==0, "number of multihead is wrong"
        self.d_m_atn = d // h
        self.h = h
        self.linear_q =nn.Linear(d, d)
        self.linear_k =nn.Linear(d, d)
        self.linear_v =nn.Linear(d, d)
        self.linear_last =nn.Linear(d, d)

    
    def forward(self, query, key, value, mask=None):
        """
        args:
          - query (torch.tensor) (B x len x d)
          - key (torch.tensor) (B x len x d)
          - value (torch.tensor) (B x len x d)
          - mask (torch.tensor) (len x len)
        return:
          - out (torch.tensor) (B x len x d)
        """
        n_batch = query.shape[0]
        # m_q = self.linear_q(query).reshape(n_batch, -1, self.h, self.d_m_atn).transpose(1,2)# transposeしないと、うまく変形できない。(try.ipynb参照)
        # m_k = self.linear_k(key).reshape(n_batch, -1, self.h, self.d_m_atn).transpose(1,2)
        # m_v = self.linear_v(value).reshape(n_batch, -1, self.h, self.d_m_atn).transpose(1,2)

        m_q = self.linear_q(query).view(n_batch, -1, self.h, self.d_m_atn).transpose(1,2)# transposeしないと、うまく変形できない。(try.ipynb参照)
        m_k = self.linear_k(key).view(n_batch, -1, self.h, self.d_m_atn).transpose(1,2)
        m_v = self.linear_v(value).view(n_batch, -1, self.h, self.d_m_atn).transpose(1,2)
        z = []
        for i in range(self.h):
            z.append(attention(m_q[:,i,:,:], m_k[:,i,:,:], m_v[:,i,:,:],mask))
        z = torch.cat(z, 2)
        return self.linear_last(z)


class Decoder(nn.Module):
    """
    エンコーダ全体
    embedded data -> decoder_layer -> x N -> hidden data(to output)

    args:
      - N (int) : encoder_layerの層数
    """
    def __init__(self, d_model, N_dec, h_dec):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(N_dec):
            self.layers.append(DecoderLayer(d_model, h_dec))
    
    def forward(self, y_emb, memory, mask):
        """
        args:
          - x_emb (torch.tensor) (B x len x d) : 位置エンコーディングまで終わったもの
        """
        for layer in self.layers:
            y_emb = layer(y_emb, memory, mask)
        return y_emb

class DecoderLayer(nn.Module):
    """
    エンコーダのそれぞれのブロック
    input -> (residual path) -> masked Multihead attention -> add&norm -> (residual path) 
        -> Multihead attention -> add&norm -> (residual path) -> feed forward -> add&norm -> out

    args:
      - feature : [B, len, d]
    """
    def __init__(self, d_model, h=8, dropout=0.0):
        super().__init__()
        self.masked_multi_attention = MultiHeadAttention(h,d_model,dropout)
        self.multi_attention = MultiHeadAttention(h,d_model,dropout)
        self.feedforward = FeedForward(d_model)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.layer_norm_3 = LayerNorm(d_model)
    
    def forward(self, x, memory, mask):
        """
        args:
          - x : output of previous layer
          - memory : output of encoder
        """
        out = self.masked_multi_attention(x,x,x,mask)
        out = self.layer_norm_1(x + out)
        res = out
        out = self.multi_attention(out, memory, memory)
        out = self.layer_norm_2(res + out)
        res = out
        out = self.feedforward(out)
        out = self.layer_norm_3(res + out)
        return out

class Embedding(nn.Module):
    """
    input と output で別のインスタンスにする必要がある
    args:
      - vocab_num (int) : 語彙数
      - d_model (int) : 埋め込みベクトルの次元数
    """
    def __init__(self, vocab_num, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_num, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    位置エンコーディング
    
    args:
      - d_model (int) : ベクトルの次元数
      - dropout (float)
      - device
      - max_len (int) : 許容しうる最大の長さの文章
    """
    def __init__(self, d_model, dropout, device="cpu", max_len = 10000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).type(torch.float32)
        tmp = torch.arange(0,d_model,2)
        den = 1/torch.pow(torch.ones(int(d_model/2))*10000,2*tmp/d_model)
        den = den.unsqueeze(0)
        self.pe[:,0::2] = torch.sin(torch.matmul(pos,den))
        self.pe[:,1::2] = torch.cos(torch.matmul(pos,den))
        self.pe = self.pe.to(device)

    def forward(self, x):
        return x + self.pe[:x.shape[1],:]

class Generator(nn.Module):
    """
    最終層。モデルの出力と文字を対応させる。
    output of model -> linear -> softmax -> output probability

    args:
      - d_model (int) : 
      - vocab_num (int) : 
    """
    def __init__(self, d_model, vocab_num):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_num)
    def forward(self, x):
        # return F.softmax(self.linear(x), dim=-1)
        return self.linear(x)

class Model(nn.Module):
    """
    モデル全体
    input of model (word of id) -> embedding -> PositionalEncoding
                        -> EncoderDecoder -> Generator -> probability of words

    args:
      - device
      - d_model
      - vocab_num (int) : 全語彙数
      - dropout
      - N_enc (int) : number of encoderlayer
      - N_dec (int) : number of decoderlayer
      - h_enc (int) : number of multihead in encoder
      - h_dec (int) : number of multihead in decoder
    """
    def __init__(self, device, d_model, vocab_num, dropout, N_enc, N_dec, h_enc, h_dec):
        super().__init__()
        self.vocab_num = vocab_num
        self.emb_x = Embedding(vocab_num, d_model)
        self.pos_enc_x = PositionalEncoding(d_model, dropout, device)
        self.emb_y = Embedding(vocab_num, d_model)
        self.pos_enc_y = PositionalEncoding(d_model, dropout, device)
        self.enc_dec = EncoderDecoder(d_model, N_enc, N_dec, h_enc, h_dec)
        self.gen = Generator(d_model, vocab_num)
    
    def forward(self, x, y, mask):
        """
        args:
          - x (torch.tensor) (B x len) : それぞれの文章(id)
          - y (torch.tensor) (B x len) : それぞれの文章(id)
          - mask (torch.tensor) (len x len) : マスク(カンニング部は0で埋め、それ以外は1)
        output:
          - x (torch.tensor) (B x len) : 変換後のそれぞれの文章(id)
        """
        x = self.emb_x(x)
        x = self.pos_enc_x(x)
        y = self.emb_y(y)
        y = self.pos_enc_y(y)
        out = self.enc_dec(x, y, mask)
        out = self.gen(out)
        return out
    
    def generate(self, x, z_def=1):
        """
        自己回帰的に生成する.
        簡易的に、所望の長さになったら終了するように実装する。本来は<eos>が出るまで。
        args:
            - z_dec (int) : デコーダに入力する最初の文字
        """
        B, l= x.shape
        x = self.emb_x(x)
        x = self.pos_enc_x(x)
        z = self.enc_dec.encode(x)
        y = torch.ones(size=(B, 1)).long() * (self.vocab_num-1) # start of sequence的なやつは、文章の単語にないやつの方がいいかなと。。。」
        for i in range(l):
            mask = make_mask(y.shape[1])
            tmp_y = self.emb_y(y)
            tmp_y = self.pos_enc_y(tmp_y)
            tmp_y = self.enc_dec.decode(tmp_y, z, mask)
            tmp_y = self.gen(tmp_y)
            next_word = torch.max(tmp_y[:,-1,:],dim=-1)[1]
            y = torch.cat([y,next_word.unsqueeze(0)],dim = -1)
        return y[:,1:]
        # return y


def make_mask(word_len):
    """
    make mask of tgt data
    set 0 where you must not look, otherwise, set 1.

    # todo
    - we maybe need mask for padding, also.
    """
    mask = np.tril(np.ones((word_len,word_len))).astype(np.float32)
    mask = torch.tensor(mask)
    return mask


if __name__ == "__main__":
    d_model = 512
    # data_x = torch.tensor(np.random.rand(3,10,d_model).astype(np.float32))
    # data_y = torch.tensor(np.random.rand(3,10,d_model).astype(np.float32))
    data_x = torch.LongTensor(np.random.randint(0,10, size=(1,20)))
    data_y = torch.LongTensor(np.random.randint(0,10, size=(1,20)))
    mask = torch.tensor(np.random.randint(0,2, size=(20,20)).astype(np.float32))
    model = Model("cpu", 512, 10, 0.0, 6, 6, 8, 8)
    g = model.generate(data_x)
    print(model(data_x, data_y, mask).shape)
    print(g.shape)







