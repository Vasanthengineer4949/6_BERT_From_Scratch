from transformers import AutoConfig
import config
from data_input import DataInput
from head import ClassificationHeadedTransformer

if __name__ == "__main__":
    text = "Time flies like arrow"
    datainput = DataInput(config, text)
    model_inpt = datainput.emb_input()
    modelconfig = AutoConfig.from_pretrained(config.MODEL_CKPT)
    modelconfig.num_labels = 2 # Setting it as a binary classification problem
    model = ClassificationHeadedTransformer(modelconfig)
    classify_out = model(model_inpt.input_ids)
    print("Classification Output Shape", classify_out.size())
    print("Classification Output", classify_out)

