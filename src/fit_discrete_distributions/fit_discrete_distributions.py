import numpy
import numpy as np
import wandb
from typing import Any, Dict
import pprint

import src.globals


def fit_discrete_distributions():
    """
    Simulate a discrete distribution with num_outcomes outcomes (like an num_outcomes-sided die).
    In each iteration, we:
      1) Estimate the distribution from the current data (counts per outcome).
      2) Sample num_samples_per_iteration new data from that empirical distribution.
      3) Either 'Replace' or 'Accumulate' or 'Accumulate-Subsample' the data.

    We log some diagnostic info to Weights & Biases (wandb).
    """

    run = wandb.init(
        project="rerevisiting-model-collapse-fit-discrete-distributions",
        config=src.globals.DEFAULT_DISCRETE_DISTRIBUTION_FITTING_CONFIG,
        entity=wandb.api.default_entity,
    )

    config: Dict[str, Any] = dict(wandb.config)
    pprint.pprint(config)

    np.random.seed(config["seed"])

    num_outcomes = config["num_outcomes"]
    num_samples_per_iteration = config["num_samples_per_iteration"]
    setting = config["setting"]

    proportions = np.full(shape=(num_outcomes,), fill_value=1.0 / num_outcomes)
    initial_entropy = -np.nansum(proportions * np.log(proportions))

    # Log the initial entropy and other diagnostic info.
    wandb.log(
        {
            "Num. Outcomes": num_outcomes,
            "Num. Samples per Iteration": num_samples_per_iteration,
            "Model-Fitting Iteration": 0,
            "Setting": setting,
            "Entropy": initial_entropy,
            "Initial Entropy": initial_entropy,
            "Num. Zero Proportions": np.sum(proportions == 0),
        }
    )

    # Sample data from the initial proportions.
    data = np.random.choice(
        a=np.arange(num_outcomes),
        size=num_samples_per_iteration,
        p=proportions,
        replace=True,
    )

    iteration_idx = 1
    while True:
        # Fit (i.e., estimate) the discrete distribution from the data
        if setting in {"Accumulate", "Replace"}:
            proportions = fit_proportions_from_observations(data, num_outcomes)
        elif setting == "Accumulate-Subsample":
            # We have "accumulated" data but only use a random subsample of size num_samples_per_iteration
            # to estimate the distribution
            subsample_idx = np.random.choice(
                np.arange(len(data)),
                size=num_samples_per_iteration,
                replace=False,
            )
            subsampled_data = data[subsample_idx]
            proportions = fit_proportions_from_observations(
                subsampled_data, num_outcomes
            )
        else:
            raise ValueError(f"Unknown setting: {setting}")

        # Compute the entropy of the distribution
        # 0 * log(0) is defined as 0, not NaN. We can hack this using np.nansum.
        entropy = -np.nansum(proportions * np.log(proportions))
        if numpy.isnan(entropy):
            raise ValueError("Entropy is NaN!")
        wandb.log(
            {
                "Num. Outcomes": num_outcomes,
                "Num. Samples per Iteration": num_samples_per_iteration,
                "Model-Fitting Iteration": iteration_idx,
                "Setting": setting,
                "Entropy": entropy,
                "Initial Entropy": initial_entropy,
                "Num. Zero Proportions": np.sum(proportions == 0),
            }
        )

        # This should technically be 0 but we don't want to loop forever.
        if entropy < 1e-8:
            break

        # Create data for the next iteration.
        new_data = np.random.choice(
            a=np.arange(num_outcomes), size=num_samples_per_iteration, p=proportions
        )
        if setting == "Replace":
            # Discard old data, keep only new
            data = new_data
        elif setting in {"Accumulate", "Accumulate-Subsample"}:
            data = np.concatenate([data, new_data])
        else:
            raise ValueError(f"Unknown setting: {setting}")

        iteration_idx = iteration_idx + 1

    wandb.finish()


def fit_proportions_from_observations(
    observations: np.ndarray, num_outcomes: int
) -> np.ndarray:
    """
    Estimate the distribution over O outcomes given 'data'
    where each entry in 'data' is an integer in [0, O-1].
    Returns a vector of length O with empirical frequencies.
    """
    counts = np.bincount(observations, minlength=num_outcomes)
    total = counts.sum()
    assert total > 0, "No data to fit distribution!"
    return counts / total


if __name__ == "__main__":
    fit_discrete_distributions()
    print("Finished discrete distribution simulation!")
