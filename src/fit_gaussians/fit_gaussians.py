import numpy as np
from typing import Tuple
import wandb


def fit_mean_and_cov_from_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(data, axis=0)
    Sigma = np.cov(data, rowvar=False, bias=True)
    # If the input data has shape 1, np.cov will squeeze this out, but we do not want this.
    if data.shape[1] == 1:
        Sigma = np.reshape(Sigma, newshape=(1, 1))
    return mu, Sigma
