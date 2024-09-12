import os

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "True"

# This is needed for deterministic to work.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from datasets import Dataset, DatasetDict, load_dataset
import gc
import numpy as np
import pandas as pd
import pprint
import time
import torch
from typing import Any, Dict, List
from vllm import LLM, SamplingParams, RequestOutput
from vllm.distributed.parallel_state import destroy_model_parallel
import wandb


import src.data
import src.globals


def sample_dataset_from_model():
    num_visible_devices = torch.cuda.device_count()
    assert num_visible_devices > 0, "No CUDA devices available."
    run = wandb.init(
        project="rerevisiting-model-collapse-sampling",
        config=src.globals.DEFAULT_SAMPLE_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    pprint.pprint(wandb_config)

    # Create dataset name.
    model_name_or_path = wandb_config["model_name_or_path"]
    temperature = wandb_config["temperature"]
    max_seq_length = wandb_config["max_seq_length"]
    dataset_name = f"{model_name_or_path.split('/')[1]}_temp{temperature}_max_seq_len{max_seq_length}"
    print(f"Dataset name: {dataset_name}")

    # Create empty prompts for sampling.
    sampling_prompts = [
        "" for _ in range(wandb_config["num_prompts_per_sampling_call"])
    ]

    # Load the model.
    model = LLM(
        wandb_config["model_name_or_path"],
        dtype="bfloat16",
        enforce_eager=True,  # I don't trust Google models in FA2 and SDPA doesn't work with Gemma.
    )
    print("Loaded policy model.")

    dataset_prompts = []
    dataset_responses = []

    # Rejection sample till we get enough data.
    batch_generation_idx = 0
    while len(dataset_prompts) < wandb_config["total_num_samples"]:
        # Note: Why do we use `n>1`?
        # Answer 1: Legacy. Previously, I was passing in multiple different prompts and sampling outputs from
        # each multiple times.
        # Answer 2: I also suspect but have not verified that VLLM is friendlier to 64 prompts with
        # 64 samples per prompt, rather than 4096 prompts and 1 sample per prompt.
        policy_model_sampling_params = SamplingParams(
            n=wandb_config["num_samples_per_prompt"],
            max_tokens=wandb_config["max_seq_length"],
            ignore_eos=False,
            seed=batch_generation_idx,
            stop_token_ids=[model.get_tokenizer().eos_token_id],
            temperature=wandb_config["temperature"],
        )

        requests_outputs: List[RequestOutput] = model.generate(
            prompts=sampling_prompts, sampling_params=policy_model_sampling_params
        )

        batch_samples = [
            output.text
            for request_output in requests_outputs
            for output in request_output.outputs
        ]

        for sample in batch_samples:
            if not isinstance(sample, str):
                print()  # Debugging.

            # Only keep outputs that have both a user prompt and an assistant response.
            if not sample.startswith("user: "):
                continue
            if "assistant: " not in sample:
                continue

            split_sample = sample.split("assistant: ")
            if len(split_sample) != 2:
                continue  # Sometimes the model doesn't generate a response after "assistant: ".
            raw_prompt, raw_response = split_sample
            processed_prompt = raw_prompt.lstrip("user: ").strip()
            processed_response = raw_response.strip()
            dataset_prompts.append(processed_prompt)
            dataset_responses.append(processed_response)

        batch_generation_idx += 1
        print(f"Sampled {len(dataset_prompts)} / {wandb_config['total_num_samples']}.")

    print("Sampled outputs from model.")

    # We might have oversampled (depending on batch size). Trim back to get the
    # correct number of samples.
    dataset_prompts = dataset_prompts[: wandb_config["total_num_samples"]]
    dataset_responses = dataset_responses[: wandb_config["total_num_samples"]]

    # Create the dataset.
    dataset = Dataset.from_dict(
        {
            "prompt": dataset_prompts,
            "response": dataset_responses,
        }
    )
    dataset_dict = DatasetDict({"train": dataset})

    # Push the dataset to HuggingFace.
    print("Pushing the dataset to HuggingFace...")
    commit_info = dataset_dict.push_to_hub(repo_id=dataset_name)
    print(commit_info)
    print("Pushed the dataset to HuggingFace!")

    # Freeing up VLLM memory is harder than I thought!
    # See: https://github.com/vllm-project/vllm/issues/1908
    # Hit it with everything recommended!
    destroy_model_parallel()
    del model.llm_engine.model_executor.driver_worker
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(7)
    print("Sampled outputs from policy model.")


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )

    # dataset = load_dataset("nvidia/HelpSteer2")
    sample_dataset_from_model()

    print("Finished sample_outputs_from_policy_model.py!")
