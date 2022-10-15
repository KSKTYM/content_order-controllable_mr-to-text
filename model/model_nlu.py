#! python

import torch
import torch.nn as nn
import sys
sys.path.append('..')
from model.model_sub import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer

## Encoder
class NLU_Encoder(nn.Module):
    def __init__(self, txt_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, txt_max_num_token, device):
        super().__init__()

        self.device = device
        self.tok_embedding = nn.Embedding(txt_dim, hid_dim)
        self.pos_embedding = nn.Embedding(txt_max_num_token, hid_dim)
        self.layers = nn.ModuleList([NLU_EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, txt, txt_mask):
        # txt: [batch_size, txt_max_num_token] (128, 88)
        # txt_mask: [batch_size, 1, 1, txt_max_num_token] (128, 1, 1, 88)
        #print('NLU_Encoder(1) txt: '+str(txt.shape))
        #print('NLU_Encoder(1) txt_mask: '+str(txt_mask.shape))

        batch_size = txt.shape[0]
        txt_len = txt.shape[1]
        pos = torch.arange(0, txt_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos: [batch_size, txt_max_num_token] (128, 88)
        #print('NLU_Encoder(2) pos: '+str(pos.shape))

        txt = self.dropout((self.tok_embedding(txt) * self.scale) + self.pos_embedding(pos))
        # txt: [batch_size, txt_nax_num_token, hid_dim] (128, 88, 256)
        #print('NLU_Encoder(3) txt: '+str(txt.shape))

        for layer in self.layers:
            txt = layer(txt, txt_mask)
        # txt: [batch_size, txt_max_num_token, hid_dim] (128, 88, 256)
        #print('NLU_Encoder(4) txt: '+str(txt.shape))

        return txt


class NLU_EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, txt, txt_mask):
        # txt: [batch_size, txt_max_num_token, hid_dim] (128, 88, 256)
        # txt_mask: [batch_size, 1, 1, txt_max_num_token] (128, 1, 1, 88)
        #print('NLU_EncoderLayer(1) txt: '+str(txt.shape))
        #print('NLU_EncoderLayer(1) txt_mask: '+str(txt_mask.shape))

        _txt, _ = self.self_attention(txt, txt, txt, txt_mask)
        # _txt: [batch_size, txt_max_num_token, hid_dim] (128, 88, 256)
        #print('NLU_EncoderLayer(2) _txt: '+str(_txt.shape))

        txt = self.layer_norm(txt + self.dropout(_txt))
        # txt: [batch_size, txt_max_num_token, hid_dim] (128, 88, 256)
        #print('NLU_EncoderLayer(3) txt: '+str(txt.shape))

        _txt = self.positionwise_feedforward(txt)
        # _txt: [batch_size, txt_max_num_token, hid_dim] (128, 88, 256)
        #print('NLU_EncoderLayer(4) _txt: '+str(_txt.shape))

        txt = self.layer_norm(txt + self.dropout(_txt))
        # txt: [batch_size, txt_max_num_token, hid_dim] (128, 88, 256)
        #print('NLU_EncoderLayer(5) txt: '+str(txt.shape))

        return txt


## Decoder (variable length)
class NLU_Decoder(nn.Module):
    def __init__(self, mr_value_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, mr_num_token, device):
        super().__init__()
        
        self.device = device
        self.tok_embedding = nn.Embedding(mr_value_dim, hid_dim)
        self.pos_embedding = nn.Embedding(mr_num_token, hid_dim)
        self.layers = nn.ModuleList([NLU_DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.fc_out = nn.Linear(hid_dim, mr_value_dim)

    def forward(self, enc_txt, txt_mask, mr_value, mr_mask):
        # enc_txt: [batch_size, txt_max_num_token, hid_dim] (128, 88, 256)
        # txt_mask: [batch_size, 1, 1, txt_max_num_token] (128, 1, 1, 88)
        # mr_value: [batch_size, mr_num_token-1] (128, 9)
        # mr_mask: [batch_size, 1, mr_num_token-1, mr_num_token-1] (128, 1, 9, 9)
        #print('NLU_Decoder(1) enc_txt: '+str(enc_txt.shape))
        #print('NLU_Decoder(1) txt_mask: '+str(txt_mask.shape))
        #print('NLU_Decoder(1) mr_value: '+str(mr_value.shape))
        #print('NLU_Decoder(1) mr_mask: '+str(mr_mask.shape))

        batch_size = mr_value.shape[0]
        mr_len = mr_value.shape[1]

        pos = torch.arange(0, mr_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos: [batch_size, mr_num_token-1] (128, 9)
        #print('NLU_Decoder(2) pos: '+str(pos.shape))

        mr_value = self.dropout((self.tok_embedding(mr_value) * self.scale) + self.pos_embedding(pos))
        # mr_value: [batch_size, mr_num_token-1, hid_dim] (128, 9, 256)
        #print('NLU_Decoder(3) mr_value: '+str(mr_value.shape))

        for layer in self.layers:
            mr_value, attention = layer(enc_txt, txt_mask, mr_value, mr_mask)
        # mr_value: [batch_size, mr_num_token-1, hid_dim] (128, 9, 256)
        # attention: [batch_size, num_heads, mr_num_token-1, txt_max_num_token] (128, 8, 9, 88)
        #print('NLU_Decoder(4) mr_value: '+str(mr_value.shape))
        #print('NLU_Decoder(4) attention: '+str(attention.shape))

        mr_value = self.fc_out(mr_value)
        # mr_value: [batch_size, mr_num_token-1, mr_value_dim] (128, 9, 36)
        #print('NLU_Decoder(5) value: '+str(mr_value.shape))

        return mr_value, attention


class NLU_DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, enc_txt, txt_mask, mr, mr_mask):
        # enc_txt: [batch_size, txt_max_num_token, hid_dim] (128, 88, 256)
        # txt_mask: [batch_size, 1, 1, txt_max_num_token] (128, 1, 1, 88)
        # mr: [batch_size, mr_num_token-1, hid_dim] (128, 9, 256)
        #print('NLU_DecoderLayer(1) enc_txt: '+str(enc_txt.shape))
        #print('NLU_DecoderLayer(1) txt_mask: '+str(txt_mask.shape))
        #print('NLU_DecoderLayer(1) mr: '+str(mr.shape))

        # self-attention
        _mr, _ = self.self_attention(mr, mr, mr, mr_mask)
        # _mr: [batch_size, mr_num_token-1, hid_dim] (128, 9, 256)
        #print('NLU_DecoderLayer(2) _mr: '+str(_mr.shape))

        mr = self.layer_norm(mr + self.dropout(_mr))
        # mr: [batch_size, mr_num_token-1, hid_dim] (128, 9, 256)
        #print('NLU_DecoderLayer(3) mr: '+str(mr.shape))

        # encoder attention
        _mr, attention = self.encoder_attention(mr, enc_txt, enc_txt, txt_mask)
        # _mr: [batch_size, mr_num_token-1, hid_dim] (128, 9, 256)
        # attention: [batch_size, num_heads, mr_num_token-1, txt_max_num_token] (128, 8, 9, 88)
        #print('NLU_DecoderLayer(4) _mr: '+str(_mr.shape))
        #print('NLU_DecoderLayer(4) attention: '+str(attention.shape))

        mr = self.layer_norm(mr + self.dropout(_mr))
        # mr: [batch_size, mr_num_token-1, hid_dim] (128, 9, 256)
        #print('NLU_DecoderLayer(5) mr: '+str(mr.shape))

        # positionwise feedforward
        _mr = self.positionwise_feedforward(mr)
        # _mr: [batch_size, mr_num_token-1, hid_dim] (128, 9, 256)
        #print('NLU_DecoderLayer(6) _mr: '+str(_mr.shape))

        mr = self.layer_norm(mr + self.dropout(_mr))
        # mr: [batch_size, mr_num_token-1, hid_dim] (128, 9, 256)
        #print('NLU_DecoderLayer(7) mr: '+str(mr.shape))

        return mr, attention


## NLU model (variable length)
class NLU_Model(nn.Module):
    def __init__(self, encoder, decoder, mr_value_pad_idx, txt_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.mr_value_pad_idx = mr_value_pad_idx
        self.txt_pad_idx = txt_pad_idx
        self.device = device

    def make_mr_mask(self, mr):
        # mr: [batch_size, mr_num_token-1] (128, 9)
        #print('NLU_Model:make_mr_mask(1) mr: '+str(mr.shape))

        mr_pad_mask = (mr != self.mr_value_pad_idx).unsqueeze(1).unsqueeze(3)
        # mr_pad_mask: [batch_size, 1, mr_num_token-1, 1] (128, 1, 9, 1)
        #print('NLU_Model:make_mr_mask(2) mr_pad_mask: '+str(mr_pad_mask.shape))

        mr_len = mr.shape[1]
        mr_sub_mask = torch.tril(torch.ones((mr_len, mr_len), device = self.device)).bool()
        # mr_sub_mask: [mr_num_token-1, mr_num_token-1] (9, 9)
        #print('NLU_Model:make_mr_mask(3) mr_sub_mask: '+str(mr_sub_mask.shape))

        mr_mask = mr_pad_mask & mr_sub_mask
        # mr_mask: [batch_size, 1, mr_num_token-1, mr_num_token-1] (128, 1, 9, 9)
        #print('NLU_Model:make_mr_mask(4) mr_mask: '+str(mr_mask.shape))

        return mr_mask

    def make_txt_mask(self, txt):
        # txt: [batch_size, txt_max_num_token] (128, 88)
        #print('NLU_Model:make_txt_mask(1) txt: '+str(txt.shape))

        txt_mask = (txt != self.txt_pad_idx).unsqueeze(1).unsqueeze(2)
        # txt_mask: [batch_size, 1, 1, txt_max_num_token] (128, 1, 1, 88)
        #print('NLU_Model:make_txt_mask(2) txt_mask: '+str(txt_mask.shape))

        return txt_mask

    def forward(self, txt, mr_value):
        # txt: [batch_size, txt_max_num_token] (128, 88)
        # mr_value: [batch_size, mr_num_token-1] (128, 9)
        #print('NLU_Model(1) txt: '+str(txt.shape))
        #print('NLU_Model(1) mr_value: '+str(mr_value.shape))

        txt_mask = self.make_txt_mask(txt)
        # txt_mask: [batch_size, 1, 1, txt_max_num_token] (128, 1, 1, 88)
        #print('NLU_Model(2) txt_mask: '+str(txt_mask.shape))

        mr_mask = self.make_mr_mask(mr_value)
        # mr_mask: [batch_size, 1, mr_num_token-1, mr_num_token-1] (128, 1, 9, 9)
        #print('NLU_Model(3) mr_mask: '+str(mr_mask.shape))

        enc_txt = self.encoder(txt, txt_mask)
        # enc_txt: [batch_size, txt_max_num_token, hid_dim] (128, 88, 256)
        #print('NLU_Model(4) enc_txt: '+str(enc_txt.shape))

        mr_value, attention = self.decoder(enc_txt, txt_mask, mr_value, mr_mask)
        # mr_value: [batch_size, mr_num_token-1, mr_value_dim] (128, 9, 36)
        # attention: [batch_size, n_head, mr_num_token-1, txt_max_num_token] (128, 8, 9, 88)
        #print('NLU_Model(5) mr_value: '+str(mr_value.shape))
        #print('NLU_Model(5) attention: '+str(attention.shape))

        return mr_value, attention
