import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from fastcore.all import *
from transformers.file_utils import (
    PaddingStrategy,
    TensorType,
    _is_tensorflow,
    _is_torch,
    _is_torch_device,
    is_tf_available,
    is_torch_available,
    to_py_obj,
    torch_required,
)
from transformers.tokenization_utils_base import (
    BatchEncoding,
    EncodedInput,
    PreTrainedTokenizerBase,
)
from transformers.utils.logging import get_logger

logger = get_logger(__name__)


@patch_to(BatchEncoding)
@torch_required
def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
    """
    Send all values to device by calling :obj:`v.to(device)` (PyTorch only).

    Args:
        device (:obj:`str` or :obj:`torch.device`): The device to put the tensors on.

    Returns:
        :class:`~transformers.BatchEncoding`: The same instance after modification.
    """

    # This check catches things like APEX blindly calling "to" on all inputs to a module
    # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
    # into a HalfTensor
    if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
        self.data = {
            k: [{subk: subv.to(device=device) for subk, subv in vv.items()} for vv in v]
            if isinstance(v, (tuple, list, L)) and isinstance(v[0], dict)
            else v.to(device=device)
            for k, v in self.data.items()
        }
    else:
        logger.warning(
            f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported."
        )
    return self


@patch_to(PreTrainedTokenizerBase)
def pad(
    self,
    encoded_inputs: Union[
        BatchEncoding,
        List[BatchEncoding],
        Dict[str, EncodedInput],
        Dict[str, List[EncodedInput]],
        List[Dict[str, EncodedInput]],
    ],
    padding: Union[bool, str, PaddingStrategy] = True,
    max_length: Optional[int] = None,
    pad_to_multiple_of: Optional[int] = None,
    return_attention_mask: Optional[bool] = None,
    return_tensors: Optional[Union[str, TensorType]] = None,
    verbose: bool = True,
    model_input_names: List = None,
):
    if model_input_names is not None:
        copy_model_input_names = self.model_input_names
        self.model_input_names = (
            [model_input_names]
            if isinstance(model_input_names, str)
            else model_input_names
        )

    if isinstance(encoded_inputs, (list, tuple)) and isinstance(
        encoded_inputs[0], (dict, BatchEncoding)
    ):
        encoded_inputs = {
            key: [example[key] for example in encoded_inputs]
            for key in encoded_inputs[0].keys()
            if key in self.model_input_names
        }

    # The model's main input name, usually `input_ids`, has be passed for padding
    if self.model_input_names[0] not in encoded_inputs:
        raise ValueError(
            "You should supply an encoding or a list of encodings to this method "
            f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
        )

    required_input = encoded_inputs[self.model_input_names[0]]

    if not required_input:
        if return_attention_mask:
            encoded_inputs["attention_mask"] = []
        return encoded_inputs

    # If we have PyTorch/TF/NumPy tensors/arrays as inputs, we cast them as python objects
    # and rebuild them afterwards if no return_tensors is specified
    # Note that we lose the specific device the tensor may be on for PyTorch

    first_element = required_input[0]
    if isinstance(first_element, (list, tuple)):
        # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
        index = 0
        while len(required_input[index]) == 0:
            index += 1
        if index < len(required_input):
            first_element = required_input[index][0]
    # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
    if not isinstance(first_element, (int, list, tuple)):
        if is_tf_available() and _is_tensorflow(first_element):
            return_tensors = "tf" if return_tensors is None else return_tensors
        elif is_torch_available() and _is_torch(first_element):
            return_tensors = "pt" if return_tensors is None else return_tensors
        elif isinstance(first_element, np.ndarray):
            return_tensors = "np" if return_tensors is None else return_tensors
        else:
            raise ValueError(
                f"type of {first_element} unknown: {type(first_element)}. "
                f"Should be one of a python, numpy, pytorch or tensorflow object."
            )

        for key, value in encoded_inputs.items():
            encoded_inputs[key] = to_py_obj(value)

    # Convert padding_strategy in PaddingStrategy
    padding_strategy, _, max_length, _ = self._get_padding_truncation_strategies(
        padding=padding, max_length=max_length, verbose=verbose
    )

    required_input = encoded_inputs[self.model_input_names[0]]
    if required_input and not isinstance(required_input[0], (list, tuple)):
        encoded_inputs = self._pad(
            encoded_inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )
        return BatchEncoding(encoded_inputs, tensor_type=return_tensors)

    batch_size = len(required_input)
    assert all(
        len(v) == batch_size for v in encoded_inputs.values()
    ), "Some items in the output dictionary have a different batch size than others."

    if padding_strategy == PaddingStrategy.LONGEST:
        max_length = max(len(inputs) for inputs in required_input)
        padding_strategy = PaddingStrategy.MAX_LENGTH

    batch_outputs = {}
    for i in range(batch_size):
        inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
        outputs = self._pad(
            inputs,
            max_length=max_length,
            padding_strategy=padding_strategy,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        for key, value in outputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []
            batch_outputs[key].append(value)

    self.model_input_names = copy_model_input_names
    return BatchEncoding(batch_outputs, tensor_type=return_tensors)


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


# def get_logger():
#     log = logging.getLogger(__name__)
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO,
#         handlers=[
#             logging.FileHandler(
#                 "{}.log".format(get_current_time()),
#                 mode="w",
#                 encoding="utf-8",
#             )
#         ],
#     )
#     return log


def mk_dir(path, parents=True, exist_ok=True):
    p = Path(path)
    p.mkdir(parents=parents, exist_ok=exist_ok)
    return p


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def load_json(fn):
    "Load a json file from a file name or opened file"
    with open_file(fn, "r", encoding="utf8") as f:
        return json.load(f)


def save_json(fn, o):
    "Save a json file, to a file name or opened file"
    with open_file(fn, "w", encoding="utf8") as f:
        return json.dump(o, f, ensure_ascii=False)


def load_csv(fn, **kwargs):
    "Load a csv file from a file name"
    kwargs.pop("sep", None)
    return pd.read_csv(fn, **kwargs)


def load_tsv(fn, **kwargs):
    "Load a tsv file from a file name"
    kwargs.pop("sep", None)
    return pd.read_csv(fn, sep="\t", **kwargs)
