# import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from typing import Any, Dict, Tuple
import wandb

import src.globals


def fit_kernel_density_estimators():
    run = wandb.init(
        project="rerevisiting-model-collapse-fit-kdes",
        config=src.globals.DEFAULT_KERNDEL_DENSITY_FITTING_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    pprint.pprint(wandb_config)

    # Set the random seed for reproducibility
    np.random.seed(wandb_config["seed"])

    assert wandb_config["setting"] in {"Replace", "Accumulate"}

    # This doesn't need to be Gaussian, but Gaussian is a fine starting point.
    init_data = create_init_data(
        num_samples_per_iteration=wandb_config["num_samples_per_iteration"],
        data_config_dict=wandb_config["data_config"],
    )
    init_data_train, init_data_test = train_test_split(init_data, test_size=0.5)
    data = init_data_train.copy()

    kde = KernelDensity(
        kernel=wandb_config["kernel"], bandwidth=wandb_config["kernel_bandwidth"]
    )

    # Iterate over the number of iterations
    for iteration_idx in range(1, wandb_config["num_iterations"] + 1):
        # Fit the data.
        kde.fit(data)

        # Score the test data.
        neg_log_prob_test = -kde.score_samples(init_data_test)
        mean_neg_log_prob_test = np.mean(neg_log_prob_test)

        # Create data for the next model-fitting iteration.
        new_data = kde.sample(n_samples=wandb_config["num_samples_per_iteration"])
        if wandb_config["setting"] == "Replace":
            data = new_data
        elif wandb_config["setting"] == "Accumulate":
            data = np.concatenate((data, new_data))

        wandb.log(
            {
                "Model-Fitting Iteration": iteration_idx,
                "Mean Negative Log Prob (Test)": mean_neg_log_prob_test,
            },
        )

    # Visualize the final KDE.
    # plt.close()

    wandb.finish()


def create_init_data(num_samples_per_iteration: int, data_config_dict: Dict[str, Any]):
    dataset_name = data_config_dict["dataset_name"]
    if dataset_name == "blobs":
        init_data = datasets.make_blobs(
            n_samples=num_samples_per_iteration,
            **data_config_dict["dataset_kwargs"],
        )[0]
    elif dataset_name == "moons":
        # We multiply by 2 because we need test data too!
        init_data = datasets.make_moons(
            n_samples=2 * num_samples_per_iteration,
            **data_config_dict["dataset_kwargs"],
        )[0]
    elif dataset_name == "circles":
        # We multiply by 2 because we need test data too!
        init_data = datasets.make_circles(
            n_samples=2 * num_samples_per_iteration,
            **data_config_dict["dataset_kwargs"],
        )[0]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return init_data


if __name__ == "__main__":
    fit_kernel_density_estimators()
    print("Finished fit_kdes.py!")
