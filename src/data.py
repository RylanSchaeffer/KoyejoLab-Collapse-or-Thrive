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
    combined_datasets_dict = {}
    for dataset_name in dataset_names:
        with PartialState().local_main_process_first():
            datasets_dict = create_dataset_for_supervised_finetuning(
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                max_length=max_length,
                remove_columns=remove_columns,
            )
            for key, value in datasets_dict.items():
                if key not in combined_datasets_dict:
                    combined_datasets_dict[key] = [value]
                else:
                    combined_datasets_dict[key].append(value)

    # Join the datasets using interleave, using the probabilities if provided.
    if "probabilities" not in data_config_dict:
        probabilities = None
    else:
        probabilities = data_config_dict["probabilities"]
    for key in combined_datasets_dict.keys():
        try:
            # Currently, interleave datasets doesn't work with torch Subsets.
            combined_datasets_dict[key] = interleave_datasets(
                datasets=combined_datasets_dict[key],
                probabilities=probabilities,
            )
        # TODO: Find a non-hacky workaround later.
        # ValueError: Expected a list of Dataset objects or a list of IterableDataset
        # objects, but element at position 0 is a Subset.
        except ValueError:
            assert len(combined_datasets_dict[key]) == 1
            combined_datasets_dict[key] = combined_datasets_dict[key][0]

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
    if dataset_name == "nvidia/HelpSteer2":
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
            raw_datasets = raw_datasets.remove_columns(
                [
                    "prompt",
                    "response",
                    "helpfulness",
                    "correctness",
                    "coherence",
                    "complexity",
                    "verbosity",
                ]
            )
        datasets_dict = {
            "train": raw_datasets["train"],
            "eval": raw_datasets["validation"],
        }
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
