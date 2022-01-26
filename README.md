# ChineseBert_pytorch
本项目主要自定义了tokenization_chinesebert_fast.py文件中的ChineseBertTokenizerFast代码。从而可以从huggingface.co调用。
```python
pretrained_tokenizer_name = "junnyu/ChineseBERT-base"
tokenizer = ChineseBertTokenizerFast.from_pretrained(pretrained_tokenizer_name)
```

# Paper
**[ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](https://arxiv.org/pdf/2106.16038.pdf)**  
*Zijun Sun, Xiaoya Li, Xiaofei Sun, Yuxian Meng, Xiang Ao, Qing He, Fei Wu and Jiwei Li*

# Install
```bash
pip install chinesebert
or
pip install git+https://github.com/JunnYu/ChineseBert_pytorch.git
```

# Usage
## mlm test.py
```python
import torch
from chinesebert import ChineseBertForMaskedLM, ChineseBertTokenizerFast, ChineseBertConfig

pretrained_tokenizer_name = "junnyu/ChineseBERT-base"
pretrained_model_name = "ShannonAI/ChineseBERT-base"

tokenizer = ChineseBertTokenizerFast.from_pretrained(pretrained_tokenizer_name)
config = ChineseBertConfig.from_pretrained(pretrained_tokenizer_name)
chinese_bert = ChineseBertForMaskedLM.from_pretrained(pretrained_model_name, config=config)

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

## data_collate test.py
```python
from transformers import DataCollatorWithPadding
from chinesebert import ChineseBertTokenizerFast
tokenizer = ChineseBertTokenizerFast.from_pretrained("junnyu/ChineseBERT-base")
collate_fn = DataCollatorWithPadding(tokenizer)
textlist = ["弗洛伊德的悲剧凸显了在美国和世界范围", "紧迫性和重要性，国际社会必须立", "那些存在严重种族主义、种族歧视", "中方对巴基斯坦开普省发"]
batch_list = [tokenizer(t) for t in textlist]
batch = collate_fn(batch_list)
print(batch.to("cuda:0"))
"""
{'input_ids': 
tensor([[ 101, 2472, 3821,  823, 2548, 4638, 2650, 1196, 1137, 3227,  749, 1762,
         5401, 1744, 1469,  686, 4518, 5745, 1741,  102],
        [ 101, 5165, 6833, 2595, 1469, 7028, 6206, 2595, 8024, 1744, 7354, 4852,
          833, 2553, 7557, 4989,  102,    0,    0,    0],
        [ 101, 6929,  763, 2100, 1762,  698, 7028, 4905, 3184,  712,  721,  510,
         4905, 3184, 3637, 6228,  102,    0,    0,    0],
        [ 101,  704, 3175, 2190, 2349, 1825, 3172, 1788, 2458, 3249, 4689, 1355,
          102,    0,    0,    0,    0,    0,    0,    0]], device='cuda:0'), 
'token_type_ids': 
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
       device='cuda:0'), 
'attention_mask': 
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]],
       device='cuda:0'), 
'pinyin_ids': 
tensor([[ 0,  0,  0,  0,  0,  0,  0,  0, 11, 26,  2,  0,  0,  0,  0,  0, 17, 26,
         20,  4,  0,  0,  0,  0, 30, 14,  1,  0,  0,  0,  0,  0,  9, 10,  2,  0,
          0,  0,  0,  0,  9, 10,  5,  0,  0,  0,  0,  0,  7, 10, 14,  1,  0,  0,
          0,  0, 15, 26,  4,  0,  0,  0,  0,  0, 25, 26,  1,  0,  0,  0,  0,  0,
         29, 14,  6, 19,  3,  0,  0,  0, 17, 10,  5,  0,  0,  0,  0,  0, 31,  6,
         14,  4,  0,  0,  0,  0, 18, 10, 14,  3,  0,  0,  0,  0, 12, 26, 20,  2,
          0,  0,  0,  0, 13, 10,  2,  0,  0,  0,  0,  0, 24, 13, 14,  4,  0,  0,
          0,  0, 15, 14, 10,  4,  0,  0,  0,  0, 11,  6, 19,  4,  0,  0,  0,  0,
         28, 10, 14,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0, 15, 14, 19,  3,  0,  0,  0,  0, 21, 20,
          4,  0,  0,  0,  0,  0, 29, 14, 19, 12,  4,  0,  0,  0, 13, 10,  2,  0,
          0,  0,  0,  0, 31, 13, 20, 19, 12,  4,  0,  0, 30,  6, 20,  4,  0,  0,
          0,  0, 29, 14, 19, 12,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         12, 26, 20,  2,  0,  0,  0,  0, 15, 14,  4,  0,  0,  0,  0,  0, 24, 13,
         10,  4,  0,  0,  0,  0, 13, 26, 14,  4,  0,  0,  0,  0,  7, 14,  4,  0,
          0,  0,  0,  0, 29, 26,  1,  0,  0,  0,  0,  0, 17, 14,  4,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0, 19, 10, 14,  4,  0,  0,  0,  0, 29, 14,
         10,  1,  0,  0,  0,  0,  8, 26, 19,  2,  0,  0,  0,  0, 31,  6, 14,  4,
          0,  0,  0,  0, 30,  6, 19,  2,  0,  0,  0,  0, 31, 13, 20, 19, 12,  4,
          0,  0, 31, 13, 20, 19, 12,  3,  0,  0, 31, 26,  2,  0,  0,  0,  0,  0,
         31, 13, 26,  3,  0,  0,  0,  0, 30, 14,  4,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0, 31, 13, 20, 19, 12,  3,  0,  0, 31, 26,  2,  0,
          0,  0,  0,  0, 22, 14,  2,  0,  0,  0,  0,  0, 24, 13, 14,  4,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0, 31, 13, 20, 19, 12,  1,  0,  0, 11,  6,
         19, 12,  1,  0,  0,  0,  9, 26, 14,  4,  0,  0,  0,  0,  7,  6,  1,  0,
          0,  0,  0,  0, 15, 14,  1,  0,  0,  0,  0,  0, 24, 14,  1,  0,  0,  0,
          0,  0, 25,  6, 19,  3,  0,  0,  0,  0, 16,  6, 14,  1,  0,  0,  0,  0,
         21, 26,  3,  0,  0,  0,  0,  0, 24, 13, 10, 19, 12,  3,  0,  0, 11,  6,
          1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]],
       device='cuda:0')}
"""
```
## tokenizer批量
```python
from chinesebert import ChineseBertTokenizerFast

tokenizer = ChineseBertTokenizerFast.from_pretrained("junnyu/ChineseBERT-base")

textlist1 = ["弗洛伊德的悲剧凸显了在美国和世界范围", "紧迫性和重要性，国际社会必须立", "那些存在严重种族主义、种族歧视", "中方对巴基斯坦开普省发"]
textlist2 = ["紧迫性和重要性，国际社会必须立", "那些存在严重种族主义、种族歧视", "中方对巴基斯坦开普省发", "弗洛伊德的悲剧凸显了在美国和世界范围"]
batch = tokenizer(textlist1,textlist2,padding=True,max_length=100,return_tensors="pt")
for k,v in batch.items():
    print(k,v.shape)
input_ids torch.Size([4, 36])
token_type_ids torch.Size([4, 36])
attention_mask torch.Size([4, 36])
pinyin_ids torch.Size([4, 288])
```

# Reference
https://github.com/ShannonAI/ChineseBert
