from collections import defaultdict

from accelerate import PartialState
from datasets import (
    concatenate_datasets,
    load_dataset,
    interleave_datasets,
    DatasetDict,
)
from functools import partial

from torch.utils.data import Dataset, random_split
from transformers import PreTrainedTokenizer
from typing import Any, Dict, List, Optional, Union


def create_datasets_for_supervised_finetuning(
    data_config_dict: Dict[str, Any],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: Optional[int] = None,
    remove_columns: bool = True,
) -> Dict[str, Union[Dataset]]:
    dataset_names: List[str] = data_config_dict["dataset"].split(",")

    # Load each dataset individually.
    combined_datasets_dict = defaultdict(list)
    for dataset_name in dataset_names:
        with PartialState().local_main_process_first():
            datasets_dict = create_dataset_for_supervised_finetuning(
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                max_length=max_length,
                remove_columns=remove_columns,
            )
            for key, value in datasets_dict.items():
                combined_datasets_dict[key].append(value)

    # Combine the datasets.
    for key in combined_datasets_dict.keys():
        combined_datasets_dict[key] = concatenate_datasets(
            dsets=combined_datasets_dict[key],
        )

    # We always want to evaluate only on the real data. Thus, overwrite the eval dataset.
    eval_dataset = create_dataset_for_supervised_finetuning(
        tokenizer=tokenizer,
        dataset_name="nvidia/HelpSteer2",
        max_length=max_length,
        remove_columns=remove_columns,
    )["eval"]
    combined_datasets_dict["eval"] = eval_dataset

    # Shuffle the datasets.
    for key in combined_datasets_dict.keys():
        combined_datasets_dict[key] = combined_datasets_dict[key].shuffle(
            seed=data_config_dict["shuffle_seed"]
        )

    return combined_datasets_dict


def create_dataset_for_supervised_finetuning(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str,
    max_length: Optional[int] = None,
    remove_columns: bool = True,
) -> Dict[str, Union[Dataset]]:
    if dataset_name == "nvidia/HelpSteer2" or "_hs2_" in dataset_name:
        raw_datasets = load_dataset(dataset_name)
        raw_datasets = raw_datasets.map(
            partial(preprocess_nvidia_helpsteer2_sft, tokenizer),
            load_from_cache_file=False,  # Always make sure we're using the latest version.
            batched=True,
            num_proc=4,
        )
        if max_length is not None:
            raw_datasets = raw_datasets.filter(
                lambda x: len(x["input_ids"]) <= max_length
            )
        if remove_columns:
            columns_to_remove = [
                col
                for col in raw_datasets["train"].column_names
                if col not in {"input_ids", "attention_mask"}
            ]
            raw_datasets = raw_datasets.remove_columns(columns_to_remove)

        datasets_dict = {
            "train": raw_datasets["train"],
        }
        # Add validation if it exists.
        if "validation" in raw_datasets:
            datasets_dict["eval"] = raw_datasets["validation"]
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset_name}")

    return datasets_dict


def preprocess_nvidia_helpsteer2_sft(
    tokenizer: PreTrainedTokenizer, examples: Dict[str, Any]
) -> Dict[str, List]:
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
    }
    for prompt, response in zip(examples["prompt"], examples["response"]):
        input_str = f"user: {prompt}\nassistant: {response}"
        tokenized_input = tokenizer(input_str)
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized_input["input_ids"].append(tokenizer.eos_token_id)
            tokenized_input["attention_mask"].append(1)
        new_examples["input_ids"].append(tokenized_input["input_ids"])
        new_examples["attention_mask"].append(tokenized_input["attention_mask"])

    return new_examples
