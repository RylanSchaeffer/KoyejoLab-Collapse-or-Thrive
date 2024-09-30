# import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
from sklearn.neighbors import KernelDensity
from typing import Any, Dict, Tuple
import wandb

import src.globals
import src.data


def fit_kernel_density_estimators():
    run = wandb.init(
        project="rerevisiting-model-collapse-fit-kdes",
        config=src.globals.DEFAULT_KERNEL_DENSITY_FITTING_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    pprint.pprint(wandb_config)

    # Set the random seed for reproducibility
    np.random.seed(wandb_config["seed"])

    setting = wandb_config["setting"]
    assert setting in {"Accumulate", "Accumulate-Subsample", "Replace"}
    num_samples_per_iteration = wandb_config["num_samples_per_iteration"]

    # This doesn't need to be Gaussian, but Gaussian is a fine starting point.
    init_data_train = src.data.create_dataset_for_kde(
        num_samples_per_iteration=wandb_config["num_samples_per_iteration"],
        data_config_dict=wandb_config["data_config"],
    )[0]
    init_data_test = src.data.create_dataset_for_kde(
        num_samples_per_iteration=500,  # Hard coded to ensure we have a large population of data for evaluation.
        data_config_dict=wandb_config["data_config"],
    )[0]
    all_data = init_data_train.copy()

    kde = KernelDensity(
        kernel=wandb_config["kernel"], bandwidth=wandb_config["kernel_bandwidth"]
    )

    # Iterate over the number of iterations
    for iteration_idx in range(1, wandb_config["num_iterations"] + 1):
        if setting in {"Accumulate", "Replace"}:
            # Fit the data.
            kde.fit(all_data)
        elif setting in {"Accumulate-Subsample"}:
            # Subsample the total data.
            subsample_idx = np.random.choice(
                np.arange(all_data.shape[0]),
                size=num_samples_per_iteration,
                replace=False,
            )
            kde.fit(all_data[subsample_idx])
        else:
            raise ValueError(f"Unknown setting: {setting}")

        # Score the test data.
        mean_neg_log_prob_test = -np.mean(kde.score_samples(init_data_test))

        # Create data for the next model-fitting iteration.
        new_data = kde.sample(n_samples=num_samples_per_iteration)
        if setting == "Replace":
            all_data = new_data
        elif setting in {"Accumulate", "Accumulate-Subsample"}:
            all_data = np.concatenate((all_data, new_data))
        else:
            raise ValueError(f"Unknown setting: {setting}")

        wandb.log(
            {
                "Model-Fitting Iteration": iteration_idx,
                "Mean Negative Log Prob (Test)": mean_neg_log_prob_test,
            },
        )

    # Visualize the final KDE.
    # plt.close()

    wandb.finish()


if __name__ == "__main__":
    fit_kernel_density_estimators()
    print("Finished fit_kdes.py!")
