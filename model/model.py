import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):

    def __init__(self, n_tokens, embedding_dim, n_heads, fc_hidden_size, n_layers, dropout_p):
        super(TransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.data_mask = None
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout_p=0)

        self.embedding = nn.Embedding(n_tokens, embedding_dim)
        encoder_layers = TransformerEncoderLayer(embedding_dim, n_heads, fc_hidden_size, dropout_p)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.fc_layer_as_decoder = nn.Linear(embedding_dim, n_tokens)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc_layer_as_decoder.weight.data.uniform_(-init_range, init_range)
        # self.fc_layer_as_decoder.bias.zero_()

    def forward(self, src):
        if self.data_mask is None or self.data_mask.size(0) != len(src):
            device = src.device
            mask = (torch.triu(torch.ones(len(src), len(src))) == 1).transpose(0, 1)
            self.data_mask = mask.to(device)

        output = self.embedding(src) * math.sqrt(self.embedding_dim)
        output = self.pos_encoder(output)
        output = self.transformer_encoder(output, self.data_mask)
        output = self.fc_layer_as_decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len = 50, dropout_p = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_p)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0,1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
