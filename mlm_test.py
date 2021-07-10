import torch
from transformers import BertConfig as GlyceBertConfig

from glycebert import GlyceBertForMaskedLM, GlyceBertTokenizerFast

pretrained_tokenizer_name = "junnyu/ChineseBERT-base"
pretrained_model_name = "ShannonAI/ChineseBERT-base"

tokenizer = GlyceBertTokenizerFast.from_pretrained(pretrained_tokenizer_name)
config = GlyceBertConfig.from_pretrained(pretrained_tokenizer_name)
chinese_bert = GlyceBertForMaskedLM.from_pretrained(
    pretrained_model_name, config=config
)

text = "北京是[MASK]国的首都。"
inputs = tokenizer(text, return_tensors="pt")
print(inputs)
maskpos = 4
with torch.no_grad():
    o = chinese_bert(**inputs)
    value, index = o.logits.softmax(-1)[0, maskpos].topk(10)

pred_tokens = tokenizer.convert_ids_to_tokens(index.tolist())
pred_values = value.tolist()

outputs = []
for t, p in zip(pred_tokens, pred_values):
    outputs.append(f"{t}|{round(p,4)}")
print(outputs)

# ['中|0.711', '我|0.2488', '祖|0.016', '法|0.0057', '美|0.0048', '全|0.0042', '韩|0.0015', '英|0.0011', '两|0.0008', '王|0.0006']
