import numpy as np
import os
import pprint
from typing import Any, Dict, Tuple
import wandb

import src.globals


def fit_gaussians():
    run = wandb.init(
        project="rerevisiting-model-collapse-fit-gaussians",
        config=src.globals.DEFAULT_GAUSSIAN_FITTING_CONFIG,
        entity=wandb.api.default_entity,
    )

    # Convert to a dictionary; otherwise, can't distribute because W&B
    # config is not pickle-able.
    wandb_config: Dict[str, Any] = dict(wandb.config)
    pprint.pprint(wandb_config)

    # Set the random seed for reproducibility
    np.random.seed(wandb_config["seed"])

    datum_dim = wandb_config["datum_dim"]
    num_samples_per_iteration = wandb_config["num_samples_per_iteration"]
    num_iterations = wandb_config["num_iterations"]
    sigma_squared = wandb_config["sigma_squared"]

    # This doesn't need to be Gaussian, but Gaussian is a fine starting point.
    init_mean = np.zeros(datum_dim)
    init_cov = sigma_squared * np.eye(datum_dim)
    initial_cov_det = np.linalg.det(init_cov)
    initial_cov_trace = np.trace(init_cov)
    init_data = np.random.multivariate_normal(
        mean=init_mean, cov=init_cov, size=num_samples_per_iteration
    )

    replaced_data = init_data.copy()
    accumulated_data = init_data.copy()

    # Iterate over the number of iterations
    for iteration_idx in range(1, num_iterations + 1):
        # Fit the mean and covariance of the two data.
        replaced_mean, replaced_cov = fit_mean_and_cov_from_data(replaced_data)
        accumulated_mean, accumulated_cov = fit_mean_and_cov_from_data(accumulated_data)

        # Compute the mean squared error of the replaced mean.
        replaced_squared_error = np.sum(np.square(replaced_mean - init_mean))
        accumulated_squared_error = np.sum(np.square(accumulated_mean - init_mean))

        # Compute the determinant of the covariance matrices.
        replaced_cov_det = np.linalg.det(replaced_cov)
        replaced_cov_trace = np.trace(replaced_cov)
        accumulated_cov_det = np.linalg.det(accumulated_cov)
        accumulated_cov_trace = np.trace(accumulated_cov)

        # Create data for the next model-fitting iteration.
        replaced_data = np.random.multivariate_normal(
            mean=replaced_mean,
            cov=replaced_cov,
            size=num_samples_per_iteration,
        )
        new_accumulated_data = np.random.multivariate_normal(
            mean=accumulated_mean,
            cov=accumulated_cov,
            size=num_samples_per_iteration,
        )
        accumulated_data = np.concatenate((accumulated_data, new_accumulated_data))

        for (
            setting,
            squared_error_of_mean,
            det_of_fit_cov_over_det_of_init_cov,
            trace_of_fit_cov_over_trace_of_init_cov,
            fit_covariance,
        ) in [
            (
                "Replace",
                replaced_squared_error,
                replaced_cov_det / initial_cov_det,
                replaced_cov_trace / initial_cov_trace,
                replaced_cov,
            ),
            (
                "Accumulate",
                accumulated_squared_error,
                accumulated_cov_det / initial_cov_det,
                accumulated_cov_trace / initial_cov_trace,
                accumulated_cov,
            ),
        ]:
            wandb.log(
                {
                    "Data Dimension (d)": datum_dim,
                    "Num. Samples per Iteration (num_samples_per_iteration)": num_samples_per_iteration,
                    r"Initial Noise ($\sigma^2$)": sigma_squared,  # "sigma_squared" is the noise variance for the true data.
                    "repeat": wandb_config["seed"],
                    "Model-Fitting Iteration": iteration_idx,
                    "Setting": setting,
                    "Squared Error of Fit Mean (Numerical)": squared_error_of_mean,
                    "Det of Fit Cov / Det of Init Cov (Numerical)": det_of_fit_cov_over_det_of_init_cov,
                    "Trace of Fit Cov / Trace of Init Cov (Numerical)": trace_of_fit_cov_over_trace_of_init_cov,
                    "Fit Covariance (Numerical)": (
                        fit_covariance[0, 0] if datum_dim == 1 else np.nan
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
