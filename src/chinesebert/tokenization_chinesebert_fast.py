# -*- coding: utf-8 -*-
from pypinyin import Style, pinyin
from transformers.models.bert import BertTokenizerFast


class ChineseBertTokenizerFast(BertTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pinyin_dict = kwargs.get("pinyin_map")
        self.id2pinyin = kwargs.get("id2pinyin")
        self.pinyin2tensor = kwargs.get("pinyin2tensor")

    def __call__(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        **kwargs
    ):
        if isinstance(text, str):
            single = True
        else:
            single = False

        tokenizer_outputs = super().__call__(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=True,
            return_length=return_length,
            verbose=verbose,
            **kwargs
        )
        tokenizer_outputs["pinyin_ids"] = self.convert_text_to_pinyin_ids(
            text, tokenizer_outputs.offset_mapping, single
        )

        tokenizer_outputs.convert_to_tensors(
            tensor_type=return_tensors,
            prepend_batch_axis=return_tensors is not None and single,
        )

        if not return_offsets_mapping:
            del tokenizer_outputs["offset_mapping"]

        return tokenizer_outputs

    def convert_text_to_pinyin_ids(self, text, offset_mapping, single):
        # get pinyin of a sentence
        pinyin_list = pinyin(
            text,
            style=Style.TONE3,
            heteronym=True,
            errors=lambda x: [["not chinese"] for _ in x],
        )
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        batch_pinyin_ids = []
        if single:
            offset_mapping = [offset_mapping]

        for batch_offset in offset_mapping:
            pinyin_ids = []
            for offset in batch_offset:
                if offset[0] in pinyin_locs and offset[1] - offset[0] == 1:
                    pinyin_ids.append(pinyin_locs[offset[0]])
                else:
                    pinyin_ids.append([0] * 8)

            batch_pinyin_ids.append(pinyin_ids)

        return batch_pinyin_ids[0] if single else batch_pinyin_ids
