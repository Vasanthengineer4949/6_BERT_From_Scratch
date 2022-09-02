import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_state):
        intermediate_output = self.linear1(hidden_state)
        intermediate_output = self.gelu(intermediate_output)
        final_output = self.linear2(intermediate_output)
        final_output = self.dropout(final_output)
        return final_output