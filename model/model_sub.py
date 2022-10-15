#! python

import torch
import torch.nn as nn

########################
## sub class
########################
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        # query: [batch_size, num_token, hid_dim] (128, 78/9, 256)
        # key: [batch_size, num_token, hid_dim] (128, 78/9, 256)
        # value: [batch_size, num_token, hid_dim] (128, 78/9, 256)
        # mask: [batch_size, 1, 1/num_token, num_token] (128, 1, 1/9, 78/9)
        #print('MultiHeadAttentionLayer(1) query: '+str(query.shape))
        #print('MultiHeadAttentionLayer(1) key: '+str(key.shape))
        #print('MultiHeadAttentionLayer(1) value: '+str(value.shape))
        #if mask is not None:
            #print('MultiHeadAttentionLayer(1) mask: '+str(mask.shape))

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q: [batch_size, num_token, hid_dim] (128, 78/9, 256)
        # K: [batch_size, num_token, hid_dim] (128, 78/9, 256)
        # V: [batch_size, num_token, hid_dim] (128, 78/9, 256)
        #print('MultiHeadAttentionLayer(2) Q: '+str(Q.shape))
        #print('MultiHeadAttentionLayer(2) K: '+str(K.shape))
        #print('MultiHeadAttentionLayer(2) V: '+str(V.shape))

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q: [batch_size, num_heads, num_token, head_dim] (128, 8, 78/9, 32)
        # K: [batch_size, num_heads, num_token, head_dim] (128, 8, 78/9, 32)
        # V: [batch_size, num_heads, num_token, head_dim] (128, 8, 78/9, 32)
        #print('MultiHeadAttentionLayer(3) Q: '+str(Q.shape))
        #print('MultiHeadAttentionLayer(3) K: '+str(K.shape))
        #print('MultiHeadAttentionLayer(3) V: '+str(V.shape))

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy: [batch_size, num_heads, num_token, num_token] (128, 8, 78/9, 78/9)
        #print('MultiHeadAttentionLayer(4) energy: '+str(energy.shape))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        # energy: [batch_size, num_heads, num_token, num_token] (128, 8, 78/9, 78/9)
        #print('MultiHeadAttentionLayer(5) energy: '+str(energy.shape))

        attention = torch.softmax(energy, dim = -1)
        # attention: [batch_size, num_heads, num_token, num_token] (128, 8, 78/9, 78/9)
        #print('MultiHeadAttentionLayer(6) attention: '+str(attention.shape))

        x = torch.matmul(self.dropout(attention), V)
        # x: [batch_size, num_heads, num_token, head_dim] (128, 8, 78/9, 32)
        #print('MultiHeadAttentionLayer(7) x: '+str(x.shape))

        x = x.permute(0, 2, 1, 3).contiguous()
        # x: [batch_size, num_token, num_heads, head_dim] (128, 78/9, 8, 32)
        #print('MultiHeadAttentionLayer(8) x: '+str(x.shape))

        x = x.view(batch_size, -1, self.hid_dim)
        # x: [batch_size, num_token, hid_dim] (128, 78/9, 256)
        #print('MultiHeadAttentionLayer(9) x: '+str(x.shape))

        x = self.fc_o(x)
        # x: [batch_size, num_token, hid_dim] (128, 78/9, 256)
        #print('MultiHeadAttentionLayer(10) x: '+str(x.shape))

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, num_token, hid_dim] (128, 78/9, 256)
        #print('PositionwiseFeedforwardLayer(1) x: '+str(x.shape))

        x = self.dropout(torch.relu(self.fc_1(x)))
        # x: [batch_size, num_token, pf_dim] (128, 78/9, 512)
        #print('PositionwiseFeedforwardLayer(2) x: '+str(x.shape))

        x = self.fc_2(x)
        # x: [batch_size, num_token, hid_dim] (128, 78/9, 256)
        #print('PositionwiseFeedforwardLayer(3) x: '+str(x.shape))

        return x
