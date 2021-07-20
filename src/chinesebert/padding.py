from dataclasses import dataclass
from typing import Optional, Union

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from chinesebert.jy_utils import *


@dataclass
class DataCollatorForChineseBERT:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch = self.tokenizer.pad_new(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            model_input_names=["input_ids", "token_type_ids", "attention_mask"],
        )
        # batch_pinyin_ids
        batch_pinyin_ids = self.tokenizer.pad_new(
            features,
            padding="max_length",
            max_length=8 * batch.input_ids.size(1),
            return_tensors="pt",
            model_input_names=["pinyin_ids"],
        )
        batch["pinyin_ids"] = batch_pinyin_ids["pinyin_ids"]
        del batch_pinyin_ids

        labels = L(features).attrgot("labels").items
        if labels[0] is None:
            return batch

        batch["labels"] = torch.stack(labels, dim=0)
        return batch
