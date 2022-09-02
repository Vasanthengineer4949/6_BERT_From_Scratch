from torch import nn
from transformers import AutoConfig, AutoTokenizer

class DataInput:

    def __init__(self, config, inp_text):
        self.model_ckpt = config.MODEL_CKPT
        self.input_text = inp_text

    def emb_input(self):
        modelconfig = AutoConfig.from_pretrained(self.model_ckpt)
        token_emb = nn.Embedding(modelconfig.vocab_size, modelconfig.hidden_size)
        tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        inputs = tokenizer(self.input_text, return_tensors="pt", add_special_tokens=False)
        return inputs

        

