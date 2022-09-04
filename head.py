import torch
import torch.nn as nn
from encoder import Encoder

class ClassificationHeadedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = Encoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids):
        transformer_out = self.transformer(input_ids)[:, 0, :]
        logits = self.classifier(self.dropout(transformer_out))
        return logits