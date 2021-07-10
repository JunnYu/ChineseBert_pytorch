# GlyceBert_pytorch
本项目主要自定义了tokenization_glycebert_fast.py文件中的GlyceBertTokenizerFast代码。从而可以从huggingface.co调用。
```python
pretrained_tokenizer_name = "junnyu/ChineseBERT-base"
tokenizer = GlyceBertTokenizerFast.from_pretrained(pretrained_tokenizer_name)
```

# Paper
**[ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](https://arxiv.org/pdf/2106.16038.pdf)**  
*Zijun Sun, Xiaoya Li, Xiaofei Sun, Yuxian Meng, Xiang Ao, Qing He, Fei Wu and Jiwei Li*

# Install
```bash
pip install glycebert
or
pip install git+https://github.com/JunnYu/GlyceBert_pytorch.git
```

# Usage
```python
import torch
from transformers import BertConfig as GlyceBertConfig

from glycebert import GlyceBertForMaskedLM, GlyceBertTokenizerFast

# 使用我这个里面的tokenizer config和model config
pretrained_tokenizer_name = "junnyu/ChineseBERT-base" #"junnyu/ChineseBERT-large"
pretrained_model_name = "ShannonAI/ChineseBERT-base" #"ShannonAI/ChineseBERT-large"

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

# base  ['中|0.711', '我|0.2488', '祖|0.016', '法|0.0057', '美|0.0048', '全|0.0042', '韩|0.0015', '英|0.0011', '两|0.0008', '王|0.0006']
# large ['中|0.8341', '我|0.1479', '祖|0.0157', '全|0.0007', '国|0.0005', '帝|0.0001', '该|0.0001', '法|0.0001', '一|0.0001', '咱|0.0001']
```

# Reference
https://github.com/ShannonAI/ChineseBert
