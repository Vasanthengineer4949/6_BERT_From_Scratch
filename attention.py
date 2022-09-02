import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)

    def scaled_dot_product_attention(self, q, k, v):
        attention_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(k.size()[-1])
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_outputs = torch.bmm(attention_weights, v)
        return attention_outputs

    def forward(self, hidden_state):
        attn_out = self.scaled_dot_product_attention(
            self.query(hidden_state),
            self.key(hidden_state),
            self.value(hidden_state)
        )
        return attn_out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = 8
        self.head_dim = self.embed_dim // self.num_heads


        self.query = nn.Linear(self.embed_dim, self.head_dim)
        self.key = nn.Linear(self.embed_dim, self.head_dim)
        self.value = nn.Linear(self.embed_dim, self.head_dim)

        self.attention_heads = nn.ModuleList(
            [AttentionHead(self.embed_dim, self.head_dim) for _ in range(self.num_heads)]
        )

        self.multi_output_linear = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_state):
        multi_out = torch.cat([attn_head(hidden_state) for attn_head in self.attention_heads], dim=-1)
        multi_out = self.multi_output_linear(multi_out)
        return multi_out
