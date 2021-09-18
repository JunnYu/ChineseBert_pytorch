# -*- coding: utf-8 -*-

from functools import lru_cache

from pypinyin import Style, pinyin
from transformers.models.bert import BertTokenizerFast


class ChineseBertTokenizerFast(BertTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pinyin_dict = kwargs.get("pinyin_map")
        self.id2pinyin = kwargs.get("id2pinyin")
        self.pinyin2tensor = kwargs.get("pinyin2tensor")
        self.special_tokens_pinyin_ids = [0] * 8

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
            is_batched = False
        else:
            is_batched = True

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
        if return_overflowing_tokens:
            overflow_to_sample_mapping = tokenizer_outputs["overflow_to_sample_mapping"]
        else:
            overflow_to_sample_mapping = list(
                range(len(tokenizer_outputs["input_ids"]))
            )

        offset_mapping = tokenizer_outputs["offset_mapping"]

        if is_batched:
            if text_pair is None:
                text_pair = [None] * len(text)
            pinyin_ids = []

            for i, sample_id in enumerate(overflow_to_sample_mapping):
                pinyin_ids.append(
                    self.get_pinyin_ids(
                        text[sample_id], text_pair[sample_id], offset_mapping[i]
                    )
                )
        else:
            pinyin_ids = []
            if isinstance(offset_mapping[0], list):
                for op in offset_mapping:
                    pinyin_ids.append(self.get_pinyin_ids(text, text_pair, op))
            else:
                pinyin_ids = self.get_pinyin_ids(text, text_pair, offset_mapping)

        tokenizer_outputs["pinyin_ids"] = pinyin_ids
        tokenizer_outputs.convert_to_tensors(tensor_type=return_tensors)
        if not return_offsets_mapping:
            del tokenizer_outputs["offset_mapping"]

        return tokenizer_outputs

    @lru_cache(9999)
    def pinyin_locs_map(self, text):
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
        return pinyin_locs

    def get_pinyin_ids(self, text, text_pair=None, offset_mapping=None):

        text_pinyin_locs = self.pinyin_locs_map(text)
        if text_pair:
            text_pair_pinyin_locs = self.pinyin_locs_map(text_pair)
        else:
            text_pair_pinyin_locs = None

        pinyin_ids = []
        special_token_count = 0

        for offset in offset_mapping:
            if offset == (0, 0):
                special_token_count += 1

            if special_token_count <= 1:
                pinyin_locs_maps = text_pinyin_locs
            else:
                pinyin_locs_maps = text_pair_pinyin_locs

            if offset[1] - offset[0] != 1:
                pinyin_ids.extend([0] * 8)
                continue
            if offset[0] in pinyin_locs_maps:
                pinyin_ids.extend(pinyin_locs_maps[offset[0]])
            else:
                pinyin_ids.extend([0] * 8)

        return pinyin_ids

# # -*- coding: utf-8 -*-
# from collections import defaultdict

# import numpy as np
# from pypinyin import Style, pinyin
# from transformers.models.bert import BertTokenizerFast


# class ChineseBertTokenizerFast(BertTokenizerFast):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.pinyin_dict = kwargs.get("pinyin_map")
#         self.id2pinyin = kwargs.get("id2pinyin")
#         self.pinyin2tensor = kwargs.get("pinyin2tensor")
#         self.special_tokens_pinyin_ids = [0] * 8

#     def __call__(
#         self,
#         text,
#         text_pair=None,
#         add_special_tokens=True,
#         padding=False,
#         truncation=False,
#         max_length=None,
#         stride=0,
#         is_split_into_words=False,
#         pad_to_multiple_of=None,
#         return_tensors=None,
#         return_token_type_ids=None,
#         return_attention_mask=None,
#         return_overflowing_tokens=False,
#         return_special_tokens_mask=False,
#         return_offsets_mapping=False,
#         return_length=False,
#         verbose=True,
#         **kwargs
#     ):

#         if isinstance(text, str):
#             is_batched = False
#         else:
#             is_batched = True

#         tokenizer_outputs = super().__call__(
#             text=text,
#             text_pair=text_pair,
#             add_special_tokens=add_special_tokens,
#             padding=padding,
#             truncation=truncation,
#             max_length=max_length,
#             stride=stride,
#             is_split_into_words=is_split_into_words,
#             pad_to_multiple_of=pad_to_multiple_of,
#             return_tensors=None,
#             return_token_type_ids=return_token_type_ids,
#             return_attention_mask=return_attention_mask,
#             return_overflowing_tokens=return_overflowing_tokens,
#             return_special_tokens_mask=return_special_tokens_mask,
#             return_offsets_mapping=True,
#             return_length=return_length,
#             verbose=verbose,
#             **kwargs,
#         )
#         (
#             padding_strategy,
#             truncation_strategy,
#             max_length,
#             kwargs,
#         ) = self._get_padding_truncation_strategies(
#             padding=padding,
#             truncation=truncation,
#             max_length=max_length,
#             pad_to_multiple_of=pad_to_multiple_of,
#             verbose=verbose,
#             **kwargs,
#         )
#         padding_strategy = padding_strategy.value
#         truncation_strategy = truncation_strategy.value
#         if text_pair:
#             pinyins_text = self.get_pinyin_ids(
#                 text, tokenizer_outputs.offset_mapping1, is_batched
#             )
#             pinyins_text_pair = self.get_pinyin_ids(
#                 text_pair, tokenizer_outputs.offset_mapping2, is_batched
#             )
#         else:
#             pinyins_text = self.get_pinyin_ids(
#                 text, tokenizer_outputs.offset_mapping1, is_batched
#             )

#         if padding_strategy in ["longest", "max_length"]:
#             pad_to_max_len = (
#                 len(tokenizer_outputs["input_ids"][0])
#                 if is_batched
#                 else len(tokenizer_outputs["input_ids"])
#             )
#         else:
#             pad_to_max_len = None

#         if is_batched:
#             if text_pair:
#                 tokenizer_outputs["pinyin_ids"] = []
#                 for pinyin1, pingyin2 in zip(pinyins_text, pinyins_text_pair):
#                     tokenizer_outputs["pinyin_ids"].append(
#                         self.build_pinyin_ids_with_special_tokens(
#                             pinyin1,
#                             pingyin2,
#                             max_length=max_length,
#                             pad_to_max_len=pad_to_max_len,
#                             truncation_strategy=truncation_strategy,
#                         )
#                     )
#             else:
#                 tokenizer_outputs["pinyin_ids"] = []
#                 for pinyin1 in pinyins_text:
#                     tokenizer_outputs["pinyin_ids"].append(
#                         self.build_pinyin_ids_with_special_tokens(
#                             pinyin1,
#                             max_length=max_length,
#                             pad_to_max_len=pad_to_max_len,
#                             truncation_strategy=truncation_strategy,
#                         )
#                     )
#         else:
#             if text_pair:
#                 tokenizer_outputs[
#                     "pinyin_ids"
#                 ] = self.build_pinyin_ids_with_special_tokens(
#                     pinyins_text,
#                     pinyins_text_pair,
#                     max_length=max_length,
#                     pad_to_max_len=pad_to_max_len,
#                     truncation_strategy=truncation_strategy,
#                 )
#             else:
#                 tokenizer_outputs[
#                     "pinyin_ids"
#                 ] = self.build_pinyin_ids_with_special_tokens(
#                     pinyins_text,
#                     max_length=max_length,
#                     pad_to_max_len=pad_to_max_len,
#                     truncation_strategy=truncation_strategy,
#                 )
#         if "offset_mapping1" in tokenizer_outputs:
#             del tokenizer_outputs["offset_mapping1"]
#         if "offset_mapping2" in tokenizer_outputs:
#             del tokenizer_outputs["offset_mapping2"]

#         tokenizer_outputs.convert_to_tensors(tensor_type=return_tensors)
#         if not return_offsets_mapping:
#             del tokenizer_outputs["offset_mapping"]

#         return tokenizer_outputs

#     def build_pinyin_ids_with_special_tokens(
#         self,
#         pinyin_ids_0,
#         pinyin_ids_1=None,
#         max_length=None,
#         pad_to_max_len=None,
#         truncation_strategy="max_length",
#     ):

#         pair = bool(pinyin_ids_1 is not None)
#         len_pingyin_ids = len(pinyin_ids_0) // 8
#         len_pair_pingyin_ids = len(pinyin_ids_1) // 8 if pair else 0
#         total_len = (
#             len_pingyin_ids
#             + len_pair_pingyin_ids
#             + self.num_special_tokens_to_add(pair=pair)
#         )

#         if max_length is not None and total_len > max_length:
#             (
#                 pinyin_ids_0,
#                 pinyin_ids_1,
#                 overflowing_pinyin_ids,
#             ) = self.truncate_pinyin_ids(
#                 pinyin_ids_0,
#                 pinyin_ids_1,
#                 total_len - max_length,
#                 truncation_strategy,
#                 0,
#             )

#         if pinyin_ids_1 is None:
#             data = (
#                 self.special_tokens_pinyin_ids
#                 + pinyin_ids_0
#                 + self.special_tokens_pinyin_ids
#             )
#             if pad_to_max_len is None:
#                 return data
#             else:
#                 data = data + self.special_tokens_pinyin_ids * (
#                     pad_to_max_len - len(data) // 8
#                 )
#                 return data

#         else:
#             data = (
#                 self.special_tokens_pinyin_ids
#                 + pinyin_ids_0
#                 + self.special_tokens_pinyin_ids
#                 + pinyin_ids_1
#                 + self.special_tokens_pinyin_ids
#             )
#             if pad_to_max_len is None:
#                 return data
#             else:
#                 data = data + self.special_tokens_pinyin_ids * (
#                     pad_to_max_len - len(data) // 8
#                 )
#                 return data

#     def get_pinyin_ids(self, text, offset_mapping, is_batched=False):
#         if not is_batched:
#             text = [text]
#             offset_mapping = [offset_mapping]

#         batch_pinyin_locs = []

#         for each_text in text:
#             pinyin_list = pinyin(
#                 each_text,
#                 style=Style.TONE3,
#                 heteronym=True,
#                 errors=lambda x: [["not chinese"] for _ in x],
#             )
#             pinyin_locs = {}
#             # get pinyin of each location
#             for index, item in enumerate(pinyin_list):
#                 pinyin_string = item[0]
#                 # not a Chinese character, pass
#                 if pinyin_string == "not chinese":
#                     continue
#                 if pinyin_string in self.pinyin2tensor:
#                     pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
#                 else:
#                     ids = [0] * 8
#                     for i, p in enumerate(pinyin_string):
#                         if p not in self.pinyin_dict["char2idx"]:
#                             ids = [0] * 8
#                             break
#                         ids[i] = self.pinyin_dict["char2idx"][p]
#                     pinyin_locs[index] = ids
#             batch_pinyin_locs.append(pinyin_locs)

#         batch_pinyin_ids = []

#         for offset_map, pinyin_locs in zip(offset_mapping, batch_pinyin_locs):
#             pinyin_ids = []
#             for offset in offset_map:
#                 if offset[1] - offset[0] != 1:
#                     pinyin_ids.extend([0] * 8)
#                     continue
#                 if offset[0] in pinyin_locs:
#                     pinyin_ids.extend(pinyin_locs[offset[0]])
#                 else:
#                     pinyin_ids.extend([0] * 8)

#             batch_pinyin_ids.append(pinyin_ids)

#         return batch_pinyin_ids if is_batched else batch_pinyin_ids[0]

#     def _convert_encoding(
#         self,
#         encoding,
#         return_token_type_ids=None,
#         return_attention_mask=None,
#         return_overflowing_tokens: bool = False,
#         return_special_tokens_mask: bool = False,
#         return_offsets_mapping: bool = False,
#         return_length: bool = False,
#         verbose: bool = True,
#     ):
#         if return_token_type_ids is None:
#             return_token_type_ids = "token_type_ids" in self.model_input_names
#         if return_attention_mask is None:
#             return_attention_mask = "attention_mask" in self.model_input_names

#         if return_overflowing_tokens and encoding.overflowing is not None:
#             encodings = [encoding] + encoding.overflowing
#         else:
#             encodings = [encoding]

#         encoding_dict = defaultdict(list)

#         for e in encodings:
#             encoding_dict["input_ids"].append(e.ids)

#             if return_token_type_ids:
#                 encoding_dict["token_type_ids"].append(e.type_ids)
#             if return_attention_mask:
#                 encoding_dict["attention_mask"].append(e.attention_mask)
#             if return_special_tokens_mask:
#                 encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
#             if return_offsets_mapping:
#                 encoding_dict["offset_mapping"].append(e.offsets)
#                 index = np.where(np.array(e.special_tokens_mask) == 1)[0]
#                 encoding_dict["offset_mapping1"].append(
#                     e.offsets[index[0] + 1 : index[1]]
#                 )
#                 if len(index) > 2:
#                     encoding_dict["offset_mapping2"].append(
#                         e.offsets[index[1] + 1 : index[2]]
#                     )

#             if return_length:
#                 encoding_dict["length"].append(len(e.ids))

#         return encoding_dict, encodings

#     def truncate_pinyin_ids(
#         self,
#         ids,
#         pair_ids=None,
#         num_tokens_to_remove: int = 0,
#         truncation_strategy="longest_first",
#         stride: int = 0,
#     ):
#         if num_tokens_to_remove <= 0:
#             return ids, pair_ids, []

#         overflowing_pinyin_ids = []
#         if truncation_strategy == "longest_first":
#             for _ in range(num_tokens_to_remove):
#                 if pair_ids is None or (len(ids) // 8) > (len(pair_ids) // 8):
#                     if not overflowing_pinyin_ids:
#                         window_len = min((len(ids) // 8), stride + 1)
#                     else:
#                         window_len = 1
#                     overflowing_pinyin_ids.extend(ids[-window_len * 8 :])
#                     ids = ids[:-8]
#                 else:
#                     if not overflowing_pinyin_ids:
#                         window_len = min((len(pair_ids) // 8), stride + 1)
#                     else:
#                         window_len = 1
#                     overflowing_pinyin_ids.extend(pair_ids[-window_len * 8 :])
#                     pair_ids = pair_ids[:-8]

#         elif truncation_strategy == "only_first":
#             if (len(ids) // 8) > num_tokens_to_remove:
#                 window_len = min((len(ids) // 8), stride + num_tokens_to_remove)
#                 overflowing_pinyin_ids = ids[-window_len * 8 :]
#                 ids = ids[: -num_tokens_to_remove * 8]

#         elif truncation_strategy == "only_second" and pair_ids is not None:
#             if (len(pair_ids) // 8) > num_tokens_to_remove:
#                 window_len = min((len(pair_ids) // 8), stride + num_tokens_to_remove)
#                 overflowing_pinyin_ids = pair_ids[-window_len * 8 :]
#                 pair_ids = pair_ids[: -num_tokens_to_remove * 8]

#         return (ids, pair_ids, overflowing_pinyin_ids)
