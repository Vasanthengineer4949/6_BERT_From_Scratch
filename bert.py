from transformers import AutoConfig
import config
from data_input import DataInput
from encoder import Encoder

if __name__ == "__main__":
    text = "Time flies like arrow"
    datainput = DataInput(config, text)
    model_inpt = datainput.emb_input()
    modelconfig = AutoConfig.from_pretrained(config.MODEL_CKPT)
    encoder = Encoder(modelconfig)
    encoder_out = encoder(model_inpt.input_ids)
    print("Encoder Output Shape", encoder_out.size())
    print("Encoder Output", encoder_out)

