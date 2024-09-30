import numpy as np
import os
import pprint
from typing import Any, Dict, Tuple
import wandb

import src.globals


def fit_gaussians():
    run = wandb.init(
        project="rerevisiting-model-collapse-fit-gaussians",
        config=src.globals.DEFAULT_MULTIVARIATE_GAUSSIAN_MODELING_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    pprint.pprint(wandb_config)

    # Set the random seed for reproducibility
    np.random.seed(wandb_config["seed"])

    data_dim = wandb_config["data_dim"]
    num_samples_per_iteration = wandb_config["num_samples_per_iteration"]
    num_iterations = wandb_config["num_iterations"]
    sigma_squared = wandb_config["sigma_squared"]
    setting = wandb_config["setting"]
    assert setting in {"Accumulate", "Accumulate-Subsample", "Replace"}

    # This doesn't need to be Gaussian, but Gaussian is a fine starting point.
    init_mean = np.zeros(data_dim)
    init_cov = sigma_squared * np.eye(data_dim)
    initial_cov_det = np.linalg.det(init_cov)
    initial_cov_trace = np.trace(init_cov)
    init_data = np.random.multivariate_normal(
        mean=init_mean, cov=init_cov, size=num_samples_per_iteration
    )

    all_data = init_data.copy()

    # Iterate over the number of iterations
    for iteration_idx in range(1, num_iterations + 1):
        # Fit the mean and covariance of the data.
        if setting in {"Accumulate", "Replace"}:
            fit_mean, fit_cov = fit_mean_and_cov_from_data(all_data)
        elif setting in {"Accumulate-Subsample"}:
            # Subsample the total data.
            subsample_idx = np.random.choice(
                np.arange(all_data.shape[0]),
                size=num_samples_per_iteration,
                replace=False,
            )
            subsample_data = all_data[subsample_idx]
            fit_mean, fit_cov = fit_mean_and_cov_from_data(subsample_data)
        else:
            raise ValueError(f"Unknown setting: {setting}")

        # Compute the squared error of the replaced mean.
        squared_error_of_fit_mean_from_init_mean = np.sum(
            np.square(fit_mean - init_mean)
        )

        # Compute the determinant of the covariance matrices.

        # Create data for the next model-fitting iteration.
        new_data = np.random.multivariate_normal(
            mean=fit_mean,
            cov=fit_cov,
            size=num_samples_per_iteration,
        )
        if setting == "Replace":
            all_data = new_data
        elif setting in {"Accumulate", "Accumulate-Subsample"}:
            all_data = np.concatenate((all_data, new_data))
        else:
            raise ValueError(f"Unknown setting: {setting}")

        wandb.log(
            {
                "Data Dimension": data_dim,
                "Num. Samples per Iteration": num_samples_per_iteration,
                "Initial Noise": sigma_squared,  # "sigma_squared" is the noise variance for the true data.
                "Model-Fitting Iteration": iteration_idx,
                "Setting": setting,
                "Squared Error of Fit Mean (Numerical)": squared_error_of_fit_mean_from_init_mean,
                "Det of Fit Cov / Det of Init Cov (Numerical)": np.linalg.det(fit_cov)
                / initial_cov_det,
                "Trace of Fit Cov / Trace of Init Cov (Numerical)": np.trace(fit_cov)
                / initial_cov_trace,
                "Fit Covariance (Numerical)": (
                    fit_cov[0, 0] if data_dim == 1 else np.nan
                ),
                "Covariance Structure": "Isotropic",
            },
        )

    wandb.finish()


def fit_mean_and_cov_from_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(data, axis=0)
    Sigma = np.cov(data, rowvar=False, bias=True)
    # If the input data has shape 1, np.cov will squeeze this out, but we do not want this.
    if data.shape[1] == 1:
        Sigma = np.reshape(Sigma, newshape=(1, 1))
    return mu, Sigma


if __name__ == "__main__":
    fit_gaussians()
    print("Finished fit_gaussians.py!")
