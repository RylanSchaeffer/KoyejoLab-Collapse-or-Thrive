# Adapted from https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py
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

import pprint
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from trl import (
    SFTConfig,
    SFTTrainer,
    get_peft_config,
)
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from typing import Any, Dict
import wandb

import src.data
import src.globals
import src.models

tqdm.pandas()


def train_supervised_finetuning():
    num_visible_devices = torch.cuda.device_count()
    assert num_visible_devices > 0, "No CUDA devices available."
    run = wandb.init(
        project="rerevisiting-model-collapse",
        config=src.globals.DEFAULT_SUPERVISED_FINETUNING_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    print("CUDA VISIBLE DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
    pprint.pprint(wandb_config)

    # Create output directory.
    sft_model_huggingface_name = create_sft_model_huggingface_name(
        wandb_config=wandb_config
    )
    print("Reward Model HuggingFace Name: ", sft_model_huggingface_name)
    output_dir = os.path.join("models", "sft", sft_model_huggingface_name)
    print("Output Directory: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    effective_global_batch_size = (
        num_visible_devices
        * wandb_config["sft_trainer_config"]["per_device_train_batch_size"]
        * wandb_config["sft_trainer_config"]["gradient_accumulation_steps"]
    )
    wandb.config.update(
        {
            "output_dir": output_dir,
            "effective_global_batch_size": effective_global_batch_size,
        },
    )

    set_seed(seed=wandb_config["seed"], deterministic=True)

    sft_trainer_config_dict: Dict[str, Any] = wandb_config["sft_trainer_config"]
    model_config_dict: Dict[str, Any] = wandb_config["model_config"]
    data_config_dict: Dict[str, Any] = wandb_config["data_config"]

    # Lightly modify configs as necessary.
    if model_config_dict["torch_dtype"] == "bfloat16":
        sft_trainer_config_dict["bf16"] = True
    else:
        sft_trainer_config_dict["bf16"] = False
    if model_config_dict["torch_dtype"] == "float16":
        sft_trainer_config_dict["fp16"] = True
    else:
        sft_trainer_config_dict["fp16"] = False
    if sft_trainer_config_dict["gradient_checkpointing"]:
        model_config_dict["use_cache"] = False
    else:
        model_config_dict["use_cache"] = True

    sft_config = SFTConfig(
        bf16=sft_trainer_config_dict["bf16"],
        data_seed=sft_trainer_config_dict["data_seed"],
        dataloader_num_workers=sft_trainer_config_dict["dataloader_num_workers"],
        dataloader_prefetch_factor=sft_trainer_config_dict[
            "dataloader_prefetch_factor"
        ],
        dataset_text_field="input_ids",
        eval_on_start=sft_trainer_config_dict["eval_on_start"],
        eval_strategy=sft_trainer_config_dict["eval_strategy"],
        eval_steps=sft_trainer_config_dict["eval_steps"],
        fp16=sft_trainer_config_dict["fp16"],
        gradient_accumulation_steps=sft_trainer_config_dict[
            "gradient_accumulation_steps"
        ],
        gradient_checkpointing=sft_trainer_config_dict["gradient_checkpointing"],
        include_num_input_tokens_seen=True,
        learning_rate=sft_trainer_config_dict["learning_rate"],
        logging_steps=sft_trainer_config_dict["logging_steps"],
        lr_scheduler_type=sft_trainer_config_dict["lr_scheduler_type"],
        max_seq_length=sft_trainer_config_dict["max_seq_length"],
        max_steps=sft_trainer_config_dict["max_steps"],
        num_train_epochs=sft_trainer_config_dict["num_train_epochs"],
        optim=sft_trainer_config_dict["optim"],
        output_dir=output_dir,
        per_device_eval_batch_size=sft_trainer_config_dict[
            "per_device_eval_batch_size"
        ],
        per_device_train_batch_size=sft_trainer_config_dict[
            "per_device_train_batch_size"
        ],
        remove_unused_columns=sft_trainer_config_dict["remove_unused_columns"],
        run_name=wandb.run.id,
        report_to=sft_trainer_config_dict["report_to"],
        save_strategy=sft_trainer_config_dict["save_strategy"],
        save_total_limit=sft_trainer_config_dict["save_total_limit"],
        seed=wandb_config["seed"],
        warmup_ratio=sft_trainer_config_dict["warmup_ratio"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config_dict["model_name_or_path"], use_fast=True, trust_remote_code=True
    )

    # trl/trainer/sft_trainer.py:408: UserWarning: You passed a tokenizer with padding_side not equal
    # to right to the SFTTrainer. This might lead to some unexpected behaviour due to overflow issues
    # when training a model in half-precision. You might consider adding tokenizer.padding_side = 'right' to your code.
    tokenizer.padding_side = "right"

    datasets_dict = src.data.create_datasets_for_supervised_finetuning(
        tokenizer=tokenizer,
        data_config_dict=data_config_dict,
        max_length=sft_config.max_seq_length,
    )
    train_dataset = datasets_dict["train"]
    eval_dataset = datasets_dict["eval"]

    model = src.models.create_model_automodelforcausallm(
        model_config_dict=model_config_dict,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    print("Finished training. Beginning final evaluation...")
    metrics = trainer.evaluate()
    trainer.log_metrics(split="eval", metrics=metrics)
    pprint.pprint(metrics)

    print(f"Finished final evaluation. Pushing to HuggingFace...")
    trainer.push_to_hub()
    # trainer.save_model(output_dir=sft_config.output_dir)
    print("Pushed to HuggingFace.")
    wandb.finish()


def create_sft_model_huggingface_name(wandb_config: Dict[str, Any]) -> str:
    simplified_base_lm_name = (
        wandb_config["model_config"]["model_name_or_path"]
        .replace("RylanSchaeffer/", "")
        .replace("google/", "")
        .replace("/", "_")
    )
    reward_model_huggingface_name = f"collapse_{simplified_base_lm_name}"
    dataset_name = wandb_config["data_config"]["dataset"]
    if dataset_name == "nvidia/HelpSteer2":
        simplified_dataset_name = "hs2"
    else:
        raise NotImplementedError
    reward_model_huggingface_name += f"_{simplified_dataset_name}"
    reward_model_huggingface_name += f"_sftsd{wandb_config['seed']}"

    # If no trailing "iterN" is present, add it.
    if "_iter" not in reward_model_huggingface_name:
        reward_model_huggingface_name += "_iter1"
    else:
        # If "iterN" is present, increment N by 1.
        reward_model_huggingface_name = reward_model_huggingface_name.replace(
            "iter",
            f"iter{int(reward_model_huggingface_name.split('iter')[-1]) + 1}",
        )

    if len(reward_model_huggingface_name) > 94:
        raise ValueError(
            f"reward_model_huggingface_name is too long: {reward_model_huggingface_name}"
        )
    return reward_model_huggingface_name


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(torch.cuda.device_count())]
        )
    train_supervised_finetuning()
    print("Finished train_supervised_finetuning.py!")
