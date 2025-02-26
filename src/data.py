from collections import defaultdict
import numpy as np
from accelerate import PartialState
from datasets import (
    concatenate_datasets,
    load_dataset,
    interleave_datasets,
    DatasetDict,
)
from functools import partial

from torch.utils.data import Dataset, Subset, random_split
from transformers import PreTrainedTokenizer
from typing import Any, Dict, List, Optional, Union


def create_datasets_dict(
    training_stage: str,
    data_config_dict: Dict[str, Any],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: Optional[int] = None,
    remove_columns: bool = True,
) -> Dict[str, Union[Dataset]]:
    assert training_stage in {"pretrain", "sft"}
    if training_stage == "pretrain":
        create_dataset_fn = create_dataset_for_pretraining
    elif training_stage == "sft":
        create_dataset_fn = create_dataset_for_supervised_finetuning
    else:
        raise ValueError(f"Invalid training stage: {training_stage}")

    dataset_names: List[str] = data_config_dict["dataset"].split(",")

    # Load each dataset individually.
    combined_datasets_dict = defaultdict(list)
    for dataset_name in dataset_names:
        with PartialState().local_main_process_first():
            datasets_dict = create_dataset_fn(
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                max_length=max_length,
                remove_columns=remove_columns,
            )
            # key and value are the train and test data
            for key, value in datasets_dict.items():
                combined_datasets_dict[key].append(value)

    # Combine the datasets.
    for key in combined_datasets_dict.keys():
        combined_datasets_dict[key] = concatenate_datasets(
            dsets=combined_datasets_dict[key],
        )

    # Shuffle the datasets.
    for key in combined_datasets_dict.keys():
        combined_datasets_dict[key] = combined_datasets_dict[key].shuffle(
            seed=data_config_dict["shuffle_seed"]
        )

    return combined_datasets_dict


def create_mixed_datasets_for_supervised_finetuning(
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
            # keys indicate train and validation data
            if dataset_name == "nvidia/HelpSteer2":
                train_data = datasets_dict["train"]
                indices = np.arange(len(train_data))[: data_config_dict["num_real"]]
                datasets_dict["train"] = train_data.select(indices)
                # print(datasets_dict['train'])
            else:
                train_data = datasets_dict["train"]
                indices = np.arange(len(train_data))[
                    : data_config_dict["num_synthetic"]
                ]
                datasets_dict["train"] = train_data.select(indices)
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
    print("the lengths of the final datasets are:")
    print(len(combined_datasets_dict["train"]))
    print(len(combined_datasets_dict["test"]))
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


def create_dataset_for_kde(
    num_samples_per_iteration: int, data_config_dict: Dict[str, Any]
):
    dataset_name = data_config_dict["dataset_name"]

    if dataset_name == "blobs":
        from sklearn import datasets

        init_data = datasets.make_blobs(
            n_samples=num_samples_per_iteration,
            **data_config_dict["dataset_kwargs"],
        )
    elif dataset_name == "circles":
        from sklearn import datasets

        init_data = datasets.make_circles(
            n_samples=num_samples_per_iteration,
            **data_config_dict["dataset_kwargs"],
        )
    elif dataset_name == "moons":
        from sklearn import datasets

        init_data = datasets.make_moons(
            n_samples=num_samples_per_iteration,
            **data_config_dict["dataset_kwargs"],
        )
    elif dataset_name == "swiss_roll":
        from sklearn import datasets

        # We multiply by 2 because we need test data too!
        init_data = datasets.make_swiss_roll(
            n_samples=num_samples_per_iteration,
            **data_config_dict["dataset_kwargs"],
        )
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return init_data


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


def preprocess_tinystories_pretrain(
    tokenizer: PreTrainedTokenizer, examples: Dict[str, Any]
) -> Dict[str, List]:
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
    }
    for text in examples["text"]:
        tokenized_input = tokenizer(text)
        # Make certain we end on EOS. See: https://arxiv.org/abs/2403.17031
        if tokenized_input["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized_input["input_ids"].append(tokenizer.eos_token_id)
            tokenized_input["attention_mask"].append(1)
        new_examples["input_ids"].append(tokenized_input["input_ids"])
        new_examples["attention_mask"].append(tokenized_input["attention_mask"])
    return new_examples
