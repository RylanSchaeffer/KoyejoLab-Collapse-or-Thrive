import numpy as np
import os
import pprint
from typing import Any, Dict
import wandb

import src.globals
import src.data


def fit_linear_regression():
    run = wandb.init(
        project="rerevisiting-model-collapse-fit-lin-regr",
        config=src.globals.DEFAULT_LINEAR_REGRESSION_FITTING_CONFIG,
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

    data_dim = wandb_config["data_dim"]
    num_samples_per_iteration = wandb_config["num_samples_per_iteration"]
    sigma_squared = wandb_config["sigma_squared"]
    sigma = np.sqrt(sigma_squared)

    # Generate the true parameter vector w_star
    w_star = np.random.randn(data_dim)

    # Generate the design matrix X.
    X_train_init = np.random.randn(num_samples_per_iteration, data_dim)
    X_train = X_train_init.copy()
    Y_train = X_train @ w_star + sigma * np.random.randn(num_samples_per_iteration)

    X_test = np.random.randn(1000, data_dim)
    Y_test = X_test @ w_star

    # Iterate over the number of iterations
    for iteration_idx in range(1, wandb_config["num_iterations"] + 1):
        # Fit the linear regression model.
        if setting in {"Accumulate", "Replace"}:
            w_hat = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
        elif setting in {"Accumulate-Subsample"}:
            # Subsample the total data.
            subsample_idx = np.random.choice(
                np.arange(X_train.shape[0]),
                size=num_samples_per_iteration,
                replace=False,
            )
            X_train_subsample = X_train[subsample_idx]
            Y_train_subsample = Y_train[subsample_idx]
            w_hat = (
                np.linalg.inv(X_train_subsample.T @ X_train_subsample)
                @ X_train_subsample.T
                @ Y_train_subsample
            )
        else:
            raise ValueError(f"Unknown setting: {setting}")

        # Score the model on the test data.
        Y_test_hat = X_test @ w_hat
        test_mean_squared_error = np.mean((Y_test - Y_test_hat) ** 2)

        # Sample data for the next model-fitting iteration.
        Y_train_new = X_train_init @ w_hat + sigma * np.random.randn(
            num_samples_per_iteration
        )
        if setting == "Replace":
            X_train = X_train
            Y_train = Y_train_new
        elif setting in {"Accumulate", "Accumulate-Subsample"}:
            X_train = np.concatenate((X_train, X_train_init))
            Y_train = np.concatenate((Y_train, Y_train_new))
        else:
            raise ValueError(f"Unknown setting: {setting}")

        wandb.log(
            {
                "Model-Fitting Iteration": iteration_idx,
                "Mean Squared Error (Test)": test_mean_squared_error,
            },
        )

    wandb.finish()


if __name__ == "__main__":
    fit_linear_regression()
    print("Finished fit_linear_regression.py!")
