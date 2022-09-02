import torch
import torch.nn as nn

from attention import MultiHeadAttention
from ffn import FeedForwardNN
from embedding import Embedding

class EncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForwardNN(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_embs):
        hidden_state = self.layer_norm1(input_embs)
        attention_out = input_embs + self.multi_head_attention(hidden_state)
        print("Attention Completed")
        encoder_out = attention_out + self.layer_norm2(self.feed_forward(attention_out))
        print("Encoder Layer Done")
        return encoder_out

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        print("Embeddings Done")
        encoder_out = embeddings
        for layer in self.layers:
            encoder_out = layer(encoder_out)
        return encoder_out
