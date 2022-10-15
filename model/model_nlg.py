#! python

import torch
import torch.nn as nn
import sys
sys.path.append('..')
from model.model_sub import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer

## Encoder (variable length)
class NLG_Encoder(nn.Module):
    def __init__(self, mr_value_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, mr_num_token, device):
        super().__init__()

        self.device = device
        self.mr_value_embedding = nn.Embedding(mr_value_dim, hid_dim)
        self.pos_embedding = nn.Embedding(mr_num_token, hid_dim)
        self.layers = nn.ModuleList([NLG_EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, mr_value, mr_mask):
        # mr_value: [batch_size, mr_num_token] (128, 10)
        # mr_mask: [batch_size, 1, 1, mr_num_token] (128, 1, 1, 10)
        #print('NLG_Encoder(1) mr_value: '+str(mr_value.shape))
        #print('NLG_Encoder(1) mr_mask: '+str(mr_mask.shape))

        batch_size = mr_value.shape[0]
        mr_len = mr_value.shape[1]
        pos = torch.arange(0, mr_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos: [batch_size, mr_num_token] (128, 10)
        #print('NLG_Encoder(2) pos: '+str(pos.shape))

        mr_value = self.dropout((self.mr_value_embedding(mr_value) * self.scale) + self.pos_embedding(pos))
        # mr_value: [batch_size, mr_num_token, hid_dim] (128, 10, 256)
        #print('NLG_Encoder(3) mr_value: '+str(mr_value.shape))

        for layer in self.layers:
            mr_value = layer(mr_value, mr_mask)
        # mr_value: [batch_size, mr_num_token-1, hid_dim] (128, 10, 256)
        #print('NLG_Encoder(4) mr_value: '+str(mr_value.shape))

        return mr_value


class NLG_EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, mr, mr_mask):
        # mr: [batch_size, mr_num_token, hid_dim] (128, 10, 256)
        #print('NLG_EncoderLayer(1) mr: '+str(mr.shape))

        _mr, _ = self.self_attention(mr, mr, mr, mr_mask)
        # _mr: [batch_size, mr_num_token, hid_dim] (128, 10, 256)
        #print('NLG_EncoderLayer(2) _mr: '+str(_mr.shape))

        mr = self.layer_norm(mr + self.dropout(_mr))
        # mr: [batch_size, mr_num_token, hid_dim] (128, 10, 256)
        #print('NLG_EncoderLayer(3) mr: '+str(mr.shape))

        _mr = self.positionwise_feedforward(mr)
        # _mr: [batch_size, mr_num_token, hid_dim] (128, 10, 256)
        #print('NLG_EncoderLayer(4) _mr: '+str(_mr.shape))

        mr = self.layer_norm(mr + self.dropout(_mr))
        # mr: [batch_size, mr_num_token, hid_dim] (128, 10, 256)
        #print('NLG_EncoderLayer(5) mr: '+str(mr.shape))

        return mr


## Decoder
class NLG_Decoder(nn.Module):
    def __init__(self, txt_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, txt_max_num_token, device):
        super().__init__()
        
        self.device = device
        self.txt_embedding = nn.Embedding(txt_dim, hid_dim)
        self.pos_embedding = nn.Embedding(txt_max_num_token, hid_dim)
        self.layers = nn.ModuleList([NLG_DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.fc_out = nn.Linear(hid_dim, txt_dim)

    def forward(self, enc_mr, mr_mask, txt, txt_mask):
        # enc_mr: [batch_size, mr_num_token, hid_dim] (128, 10, 256)
        # mr_mask: [batch_size, 1, 1, mr_num_token] (128, 1, 1, 10)
        # txt: [batch_size, txt_max_num_token-1] (128, 87)
        # txt_mask: [batch_size, 1, txt_max_num_token-1, txt_max_num_token-1] (128, 1, 87, 87)
        #print('NLG_Decoder(1) enc_mr: '+str(enc_mr.shape))
        #print('NLG_Decoder(1) mr_mask: '+str(mr_mask.shape))
        #print('NLG_Decoder(1) txt: '+str(txt.shape))
        #print('NLG_Decoder(1) txt_mask: '+str(txt_mask.shape))

        batch_size = txt.shape[0]
        txt_len = txt.shape[1]
        pos = torch.arange(0, txt_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos: [batch_size, txt_max_num_token-1] (128, 87)
        #print('NLG_Decoder(2) pos: '+str(pos.shape))

        txt = self.dropout((self.txt_embedding(txt) * self.scale) + self.pos_embedding(pos))
        # txt: [batch_size, txt_max_num_token-1, hid_dim] (128, 87, 256)
        #print('NLG_Decoder(3) txt: '+str(txt.shape))

        for layer in self.layers:
            txt, attention = layer(enc_mr, mr_mask, txt, txt_mask)
        # txt: [batch_size, txt_max_num_token-1, hid_dim] (128, 87, 256)
        # attention: [batch_size, num_heads, txt_max_num_token-1, mr_num_token] (128, 8, 87, 10)
        #print('NLG_Decoder(4) txt: '+str(txt.shape))
        #print('NLG_Decoder(4) attention: '+str(attention.shape))

        txt = self.fc_out(txt)
        # txt: [batch_size, txt_max_num_token-1, txt_dim] (128, 87, 2875)
        #print('NLG_Decoder(5) txt: '+str(txt.shape))

        return txt, attention


class NLG_DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, enc_mr, mr_mask, txt, txt_mask):
        # enc_mr: [batch_size, mr_num_token, hid_dim] (128, 8, 256)
        # txt: [batch_size, txt_max_num_token-1, hid_dim] (128, 87, 256)
        # txt_mask: [batch_size, 1, txt_max_num_token-1, txt_max_num_token-1] (128, 1, 87, 87)
        #print('NLG_DecoderLayer(1) enc_mr: '+str(enc_mr.shape))
        #print('NLG_DecoderLayer(1) txt: '+str(txt.shape))
        #print('NLG_DecoderLayer(1) txt_mask: '+str(txt_mask.shape))

        _txt, _ = self.self_attention(txt, txt, txt, txt_mask)
        # _txt: [batch_size, txt_max_num_token-1, hid_dim] (128, 87, 256)
        #print('NLG_DecoderLayer(2) _txt: '+str(_txt.shape))

        txt = self.layer_norm(txt + self.dropout(_txt))
        # txt: [batch_size, txt_max_num_token-1, hid_dim] (128, 87, 256)
        #print('NLG_DecoderLayer(3) txt: '+str(txt.shape))

        _txt, attention = self.encoder_attention(txt, enc_mr, enc_mr, mr_mask)
        # _txt: [batch_size, txt_max_num_token-1, hid_dim] (128, 87, 256)
        # attention: [batch_size, num_heads, txt_max_num_token-1, mr_num_token] (128, 8, 87, 8)
        #print('NLG_DecoderLayer(4) _txt: '+str(_txt.shape))
        #print('NLG_DecoderLayer(4) attention: '+str(attention.shape))

        txt = self.layer_norm(txt + self.dropout(_txt))
        # txt: [batch_size, txt_max_num_token-1, hid_dim] (128, 87, 256)
        #print('NLG_DecoderLayer(5) txt: '+str(txt.shape))

        _txt = self.positionwise_feedforward(txt)
        # _txt: [batch_size, txt_max_num_token-1, hid_dim] (128, 87, 256)
        #print('NLG_DecoderLayer(6) _txt: '+str(_txt.shape))

        txt = self.layer_norm(txt + self.dropout(_txt))
        # txt: [batch_size, txt_max_num_token-1, hid_dim] (128, 87, 256)
        #print('NLG_DecoderLayer(7) txt: '+str(txt.shape))

        return txt, attention


## NLG model (value)
class NLG_Model(nn.Module):
    def __init__(self, encoder, decoder, mr_value_pad_idx, txt_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.mr_value_pad_idx = mr_value_pad_idx
        self.txt_pad_idx = txt_pad_idx
        self.device = device
        
    def make_mr_mask(self, mr):
        # mr: [batch_size, mr_max_num_token] (128, 10)
        #print('NLG_Model:make_mr_mask(1) mr: '+str(mr.shape))

        mr_mask = (mr != self.mr_value_pad_idx).unsqueeze(1).unsqueeze(2)
        # mr_mask: [batch_size, 1, 1, mr_max_num_token] (128, 1, 1, 10)
        #print('NLG_Model:make_mr_mask(2) mr_mask: '+str(mr_mask.shape))

        return mr_mask

    def make_txt_mask(self, txt):
        # txt: [batch_size, txt_max_num_token-1] (128, 87)
        #print('NLG_Model:make_txt_mask(1) txt: '+str(txt.shape))

        txt_pad_mask = (txt != self.txt_pad_idx).unsqueeze(1).unsqueeze(3)
        # txt_pad_mask: [batch_size, 1, txt_max_num_token-1, 1] (128, 1, 87, 1)
        #print('NLG_Model:make_txt_mask(2) txt_pad_mask: '+str(txt_pad_mask.shape))

        txt_len = txt.shape[1]
        txt_sub_mask = torch.tril(torch.ones((txt_len, txt_len), device = self.device)).bool()
        # txt_sub_mask: [txt_max_num_token-1, txt_max_num_token-1] (87, 87)
        #print('NLG_Model:make_txt_mask(3) txt_sub_mask: '+str(txt_sub_mask.shape))

        txt_mask = txt_pad_mask & txt_sub_mask
        # txt_mask: [batch_size, 1, txt_max_num_token-1, txt_max_num_token-1] (128, 1, 87, 87)
        #print('NLG_Model:make_txt_mask(4) txt_mask: '+str(txt_mask.shape))

        return txt_mask

    def forward(self, mr_value, txt):
        # mr_value: [batch_size, mr_max_num_token] (128, 10)
        # txt: [batch_size, txt_max_num_token-1] (128, 87)
        #print('NLG_Model(1) mr_value: '+str(mr_value.shape))
        #print('NLG_Model(1) txt: '+str(txt.shape))

        mr_mask = self.make_mr_mask(mr_value)
        # mr_mask: [batch_size, 1, 1, mr_max_num_token] (128, 1, 1, 10)
        #print('NLG_Model(2) mr_mask: '+str(mr_mask.shape))

        txt_mask = self.make_txt_mask(txt)
        # txt_mask: [batch_size, 1, txt_max_num_token-1, txt_max_num_token-1] (128, 1, 87, 87)
        #print('NLG_Model(3) txt_mask: '+str(txt_mask.shape))

        enc_mr = self.encoder(mr_value, mr_mask)
        # enc_mr: [batch_size, mr_max_num_token, hid_dim] (128, 10, 256)
        #print('NLG_Model(4) enc_mr: '+str(enc_mr.shape))

        txt, attention = self.decoder(enc_mr, mr_mask, txt, txt_mask)
        # txt: [batch_size, txt_max_num_token-1, txt_dim] (128, 87, 2875)
        # attention: [batch_size, n_head, txt_max_num_token-1, mr_max_num_token] (128, 8, 87, 10)
        #print('NLG_Model(5) txt: '+str(txt.shape))
        #print('NLG_Model(5) attention: '+str(attention.shape))

        return txt, attention
