# copied from https://github.com/fastai/fastcore/blob/master/fastcore/basics.py
# define patch_to START
import builtins
import copy
import functools
from collections import defaultdict
from types import FunctionType, MethodType
from typing import Any, Dict, List, Optional, Tuple, Union

from tokenizers import Encoding as EncodingFast
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    PreTokenizedInputPair,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


def copy_func(f):
    "Copy a non-builtin function (NB `copy.copy` does not work for this)"
    if not isinstance(f, FunctionType):
        return copy.copy(f)
    fn = FunctionType(
        f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__
    )
    fn.__kwdefaults__ = f.__kwdefaults__
    fn.__dict__.update(f.__dict__)
    return fn


def patch_to(cls, as_prop=False, cls_method=False):
    "Decorator: add `f` to `cls`"
    if not isinstance(cls, (tuple, list)):
        cls = (cls,)

    def _inner(f):
        for c_ in cls:
            nf = copy_func(f)
            nm = f.__name__
            # `functools.update_wrapper` when passing patched function to `Pipeline`, so we do it manually
            for o in functools.WRAPPER_ASSIGNMENTS:
                setattr(nf, o, getattr(f, o))
            nf.__qualname__ = f"{c_.__name__}.{nm}"
            if cls_method:
                setattr(c_, nm, MethodType(nf, c_))
            else:
                setattr(c_, nm, property(nf) if as_prop else nf)
        # Avoid clobbering existing functions
        return globals().get(nm, builtins.__dict__.get(nm, None))

    return _inner


# define patch_to END

# patch_to PreTrainedTokenizerFast START
@patch_to(PreTrainedTokenizerFast)
def _batch_encode_plus(
    self,
    batch_text_or_text_pairs: Union[
        List[TextInput],
        List[TextInputPair],
        List[PreTokenizedInput],
        List[PreTokenizedInputPair],
    ],
    add_special_tokens: bool = True,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    max_length: Optional[int] = None,
    stride: int = 0,
    is_split_into_words: bool = False,
    pad_to_multiple_of: Optional[int] = None,
    return_tensors: Optional[str] = None,
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
) -> BatchEncoding:

    if not isinstance(batch_text_or_text_pairs, list):
        raise TypeError(
            f"batch_text_or_text_pairs has to be a list (got {type(batch_text_or_text_pairs)})"
        )

    # Set the truncation and padding strategy and restore the initial configuration
    self.set_truncation_and_padding(
        padding_strategy=padding_strategy,
        truncation_strategy=truncation_strategy,
        max_length=max_length,
        stride=stride,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    encodings = self._tokenizer.encode_batch(
        batch_text_or_text_pairs,
        add_special_tokens=add_special_tokens,
        is_pretokenized=is_split_into_words,
    )

    # Convert encoding to dict
    # `Tokens` has type: Tuple[
    #                       List[Dict[str, List[List[int]]]] or List[Dict[str, 2D-Tensor]],
    #                       List[EncodingFast]
    #                    ]
    # with nested dimensions corresponding to batch, overflows, sequence length
    tokens_and_encodings = [
        self._convert_encoding(
            encoding=encoding,
            text_or_text_pair=text_or_text_pair,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
        )
        for encoding, text_or_text_pair in zip(
            encodings, batch_text_or_text_pairs  # add batch_text_or_text_pairs
        )
    ]

    # Convert the output to have dict[list] from list[dict] and remove the additional overflows dimension
    # From (variable) shape (batch, overflows, sequence length) to ~ (batch * overflows, sequence length)
    # (we say ~ because the number of overflow varies with the example in the batch)
    #
    # To match each overflowing sample with the original sample in the batch
    # we add an overflow_to_sample_mapping array (see below)
    sanitized_tokens = {}
    for key in tokens_and_encodings[0][0].keys():
        stack = [e for item, _ in tokens_and_encodings for e in item[key]]
        sanitized_tokens[key] = stack
    sanitized_encodings = [e for _, item in tokens_and_encodings for e in item]

    # If returning overflowing tokens, we need to return a mapping
    # from the batch idx to the original sample
    if return_overflowing_tokens:
        overflow_to_sample_mapping = []
        for i, (toks, _) in enumerate(tokens_and_encodings):
            overflow_to_sample_mapping += [i] * len(toks["input_ids"])
        sanitized_tokens["overflow_to_sample_mapping"] = overflow_to_sample_mapping

    for input_ids in sanitized_tokens["input_ids"]:
        self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
    return BatchEncoding(
        sanitized_tokens, sanitized_encodings, tensor_type=return_tensors
    )


@patch_to(PreTrainedTokenizerFast)
def _convert_encoding(
    self,
    encoding: EncodingFast,
    text_or_text_pair,  # add text_or_text_pair
    return_token_type_ids: Optional[bool] = None,
    return_attention_mask: Optional[bool] = None,
    return_overflowing_tokens: bool = False,
    return_special_tokens_mask: bool = False,
    return_offsets_mapping: bool = False,
    return_length: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], List[EncodingFast]]:
    """
    Convert the encoding representation (from low-level HuggingFace tokenizer output) to a python Dict and a list
    of encodings, take care of building a batch from overflowing tokens.

    Overflowing tokens are converted to additional examples (like batches) so the output values of the dict are
    lists (overflows) of lists (tokens).

    Output shape: (overflows, sequence length)
    """
    if return_token_type_ids is None:
        return_token_type_ids = "token_type_ids" in self.model_input_names
    if return_attention_mask is None:
        return_attention_mask = "attention_mask" in self.model_input_names

    if return_overflowing_tokens and encoding.overflowing is not None:
        encodings = [encoding] + encoding.overflowing
    else:
        encodings = [encoding]

    encoding_dict = defaultdict(list)

    for e in encodings:
        encoding_dict["input_ids"].append(e.ids)
        # add custom_input_ids
        for k, v in self.custom_input_ids(text_or_text_pair, e).items():
            encoding_dict[k].append(v)
        #########
        if return_token_type_ids:
            encoding_dict["token_type_ids"].append(e.type_ids)
        if return_attention_mask:
            encoding_dict["attention_mask"].append(e.attention_mask)
        if return_special_tokens_mask:
            encoding_dict["special_tokens_mask"].append(e.special_tokens_mask)
        if return_offsets_mapping:
            encoding_dict["offset_mapping"].append(e.offsets)
        if return_length:
            encoding_dict["length"].append(len(e.ids))

    return encoding_dict, encodings


@patch_to(PreTrainedTokenizerFast)
def custom_input_ids(self, text_or_text_pair, e):
    return {}


@patch_to(PreTrainedTokenizerFast)
def _pad(
    self,
    encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
    max_length: Optional[int] = None,
    padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    pad_to_multiple_of: Optional[int] = None,
    return_attention_mask: Optional[bool] = None,
) -> dict:
    """
    Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

    Args:
        encoded_inputs: Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
        max_length: maximum length of the returned list and optionally padding length (see below).
            Will truncate by taking into account the special tokens.
        padding_strategy: PaddingStrategy to use for padding.

            - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
            - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
            - PaddingStrategy.DO_NOT_PAD: Do not pad
            The tokenizer padding sides are defined in self.padding_side:

                - 'left': pads on the left of the sequences
                - 'right': pads on the right of the sequences
        pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
            >= 7.5 (Volta).
        return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
    """
    # Load from model defaults
    if return_attention_mask is None:
        return_attention_mask = "attention_mask" in self.model_input_names

    required_input = encoded_inputs[self.model_input_names[0]]

    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = len(required_input)

    if (
        max_length is not None
        and pad_to_multiple_of is not None
        and (max_length % pad_to_multiple_of != 0)
    ):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    needs_to_be_padded = (
        padding_strategy != PaddingStrategy.DO_NOT_PAD
        and len(required_input) != max_length
    )
    # Initialize attention mask if not present.
    if return_attention_mask and "attention_mask" not in encoded_inputs:
        encoded_inputs["attention_mask"] = [1] * len(required_input)

    if needs_to_be_padded:
        difference = max_length - len(required_input)
        if self.padding_side == "right":
            if return_attention_mask:
                encoded_inputs["attention_mask"] = (
                    encoded_inputs["attention_mask"] + [0] * difference
                )
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"]
                    + [self.pad_token_type_id] * difference
                )
            # pinyin_ids
            if "pinyin_ids" in encoded_inputs:
                encoded_inputs["pinyin_ids"] = encoded_inputs["pinyin_ids"] + [0] * (
                    difference * 8
                )
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = (
                    encoded_inputs["special_tokens_mask"] + [1] * difference
                )
            encoded_inputs[self.model_input_names[0]] = (
                required_input + [self.pad_token_id] * difference
            )
        elif self.padding_side == "left":
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs[
                    "attention_mask"
                ]
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = [
                    self.pad_token_type_id
                ] * difference + encoded_inputs["token_type_ids"]
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = [
                    1
                ] * difference + encoded_inputs["special_tokens_mask"]
            encoded_inputs[self.model_input_names[0]] = [
                self.pad_token_id
            ] * difference + required_input
        else:
            raise ValueError("Invalid padding strategy:" + str(self.padding_side))

    return encoded_inputs


# patch_to PreTrainedTokenizerFast END
