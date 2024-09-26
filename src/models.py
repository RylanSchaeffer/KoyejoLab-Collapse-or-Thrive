import datasets
import numpy as np
import os
import pprint
import torch
import torch.utils.data
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    pipeline,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from trl import (
    ModelConfig,
    get_kbit_device_map,
    get_quantization_config,
)
from typing import Any, Dict, List, Optional, Tuple, Union


def create_custom_automodelforcausallm(
    model_config_dict: Dict[str, Any],
    train_dataset: datasets.Dataset,
):
    # https://github.com/RylanSchaeffer/KoyejoLab-Revisiting-Model-Collapse/blob/f570f5025259c2b7d2008b23b08fef6e765eac15/language_modeling_experiments/train_multigpu.py#L312
    # https://github.com/RylanSchaeffer/KoyejoLab-Revisiting-Model-Collapse/blob/f570f5025259c2b7d2008b23b08fef6e765eac15/language_modeling_experiments/train_multigpu.py#L426
    raise NotImplementedError


def load_automodelforcausallm(
    model_config_dict: Dict[str, Any]
) -> AutoModelForCausalLM:
    if model_config_dict["torch_dtype"] == "bfloat16":
        torch_dtype = torch.bfloat16
    elif model_config_dict["torch_dtype"] == "float16":
        torch_dtype = torch.float16
    elif model_config_dict["torch_dtype"] == "float32":
        torch_dtype = torch.float32
    else:
        raise NotImplementedError

    if "gemma" in model_config_dict["initial_model_name_or_path"]:
        # Don't use Google models with anything other than bfloat16.
        assert torch_dtype == torch.bfloat16
        # Also use eager with Gemma.
        assert model_config_dict["attn_implementation"] == "eager"

    model_kwargs = {
        "attn_implementation": "eager",
        "device_map": "auto",
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    # Add the model config to the model kwargs.
    model_kwargs.update(model_config_dict)
    # Remove the unwanted keys.
    # TypeError: Gemma2ForCausalLM.__init__() got an unexpected keyword argument 'model_name_or_path'
    model_kwargs.pop("initial_model_name_or_path")
    model_kwargs.pop("final_model_name_or_path")

    model = AutoModelForCausalLM.from_pretrained(
        model_config_dict["initial_model_name_or_path"],
        **model_kwargs,
    )

    return model
