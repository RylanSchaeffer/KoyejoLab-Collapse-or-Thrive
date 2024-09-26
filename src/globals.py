DEFAULT_KERNDEL_DENSITY_FITTING_CONFIG = {
    "data_config": {
        "dataset_name": "blobs",
        # "dataset_name": "moons",
        # "dataset_name": "swiss_roll",
        "dataset_kwargs": {
            # "noise": 0.05,
            # "n_features": 2,
        },
    },
    "kernel": "gaussian",
    "kernel_bandwidth": 0.75,
    # "kernel": "tophat",
    "num_samples_per_iteration": 89,
    "num_iterations": 100,
    "seed": 0,
    "setting": "Accumulate",
}


DEFAULT_GAUSSIAN_FITTING_CONFIG = {
    "data_dim": 7,
    "num_samples_per_iteration": 89,
    "num_iterations": 100,
    "seed": 0,
    "setting": "Accumulate",
    "sigma_squared": 1.0,
}


DEFAULT_PRETRAINING_CONFIG = {
    "data_config": {
        # "dataset": "Anthropic/hh-rlhf",
        # "dataset": "HuggingFaceH4/ultrafeedback_binarized",
        # "dataset": "nvidia/HelpSteer2",
        # "dataset": "nvidia/HelpSteer2,RylanSchaeffer/collapse_gemma-2-2b_hs2_sftsd0_iter1_temp1.0_max_seq_len512",
        # "dataset": "roneneldan/TinyStories",
        # "dataset": "RylanSchaeffer/collapse_gemma-2-2b_hs2_sftsdXXX_iter1_temp1.0_max_seq_len512",
        "shuffle_seed": 0,
    },
    "model_config": {
        "attn_implementation": "eager",
        # "attn_implementation": "flash_attention_2",
        # "final_model_name_or_path": "RylanSchaeffer/collapse_gemma-2-2b_hs2_iter1_sftsdXXX",
        "final_model_name_or_path": "RylanSchaeffer/tmp",
        # "initial_model_name_or_path": "facebook/opt-350m",
        "initial_model_name_or_path": "google/gemma-2-2b",
        # "initial_model_name_or_path": "google/gemma-2-9b",
        "torch_dtype": "bfloat16",
        # "torch_dtype": "float16",
    },
    "sft_trainer_config": {
        "data_seed": 0,
        "dataloader_drop_last": True,
        "dataloader_num_workers": 4,
        "dataloader_prefetch_factor": 4,
        # "eval_on_start": False,
        "eval_on_start": True,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": False,
        "learning_rate": 8e-6,
        # "learning_rate": 1.41e-5,
        "logging_steps": 5,
        "lr_scheduler_type": "constant_with_warmup",
        # "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
        "max_seq_length": 512,
        "max_steps": 50,
        # "max_steps": -1,
        "num_train_epochs": 1,
        "optim": "adamw_torch",
        "per_device_eval_batch_size": 16,
        # "per_device_train_batch_size": 2,
        "per_device_train_batch_size": 8,
        "remove_unused_columns": False,
        # "remove_unused_columns": True,
        "report_to": "wandb",
        "save_strategy": "no",
        "save_total_limit": 0,
        "torch_compile": False,
        "warmup_ratio": 0.025,
    },
    "seed": 0,
}


DEFAULT_SAMPLE_CONFIG = {
    "max_seq_length": 512,
    "model_name_or_path": "RylanSchaeffer/collapse_gemma-2-2b_hs2_iter2_sftsd2",
    "num_prompts_per_sampling_call": 64,
    "num_samples_per_prompt": 64,
    "temperature": 1.0,
    "total_num_samples": 20324,  # The number of samples in the HelpSteer2 dataset.
}


DEFAULT_SUPERVISED_FINETUNING_CONFIG = {
    "data_config": {
        # "dataset": "Anthropic/hh-rlhf",
        # "dataset": "HuggingFaceH4/ultrafeedback_binarized",
        "dataset": "nvidia/HelpSteer2",
        # "dataset": "nvidia/HelpSteer2,RylanSchaeffer/collapse_gemma-2-2b_hs2_sftsd0_iter1_temp1.0_max_seq_len512",
        # "dataset": "roneneldan/TinyStories",
        # "dataset": "RylanSchaeffer/collapse_gemma-2-2b_hs2_sftsdXXX_iter1_temp1.0_max_seq_len512",
        "shuffle_seed": 0,
    },
    "model_config": {
        "attn_implementation": "eager",
        # "attn_implementation": "flash_attention_2",
        # "final_model_name_or_path": "RylanSchaeffer/collapse_gemma-2-2b_hs2_iter1_sftsdXXX",
        "final_model_name_or_path": "RylanSchaeffer/tmp",
        # "initial_model_name_or_path": "facebook/opt-350m",
        "initial_model_name_or_path": "google/gemma-2-2b",
        # "initial_model_name_or_path": "google/gemma-2-9b",
        "torch_dtype": "bfloat16",
        # "torch_dtype": "float16",
    },
    "sft_trainer_config": {
        "data_seed": 0,
        "dataloader_drop_last": True,
        "dataloader_num_workers": 4,
        "dataloader_prefetch_factor": 4,
        # "eval_on_start": False,
        "eval_on_start": True,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": False,
        "learning_rate": 8e-6,
        # "learning_rate": 1.41e-5,
        "logging_steps": 5,
        "lr_scheduler_type": "constant_with_warmup",
        # "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
        "max_seq_length": 512,
        "max_steps": 50,
        # "max_steps": -1,
        "num_train_epochs": 1,
        "optim": "adamw_torch",
        "per_device_eval_batch_size": 16,
        # "per_device_train_batch_size": 2,
        "per_device_train_batch_size": 8,
        "remove_unused_columns": False,
        # "remove_unused_columns": True,
        "report_to": "wandb",
        "save_strategy": "no",
        "save_total_limit": 0,
        "torch_compile": False,
        "warmup_ratio": 0.025,
    },
    "seed": 0,
}
