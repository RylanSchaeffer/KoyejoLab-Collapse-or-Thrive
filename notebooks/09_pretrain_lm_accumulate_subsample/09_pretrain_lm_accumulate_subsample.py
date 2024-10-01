# Ported from https://github.com/RylanSchaeffer/KoyejoLab-Revisiting-Model-Collapse/blob/main/language_modeling_experiments/plot_matthias_cameraready.ipynb

import matplotlib.colors
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

import src.analyze
import src.plot


wandb_entity = "mgerstgrasser"
group_name = "tinystories_models_1"

run_groups_to_fetch = [
    "tinystories_models_1",
    # "tinystories_1_matthias_8k_1epoch_llama",
    # "tinystories_1_matthias_8k_1epoch_llama42M_snap",
    # "tinystories_1_matthias_8k_1epoch_llama125M_nlp",
    # "tinystories_5_matthias_8k_3epoch_new",
    # "tinystories_5_matthias_8k_1epoch_new",
    # "tinystories_1_matthias_8k_1epoch_subsample",
    # "tinystories_llama_firstiter",
]

from datetime import datetime
import joblib
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import wandb


import src.plot


# We have the same vocab size everywhere, so same uniform loss.
uniform_loss = math.log(8000)

# Create API to retrieve the runs from wandb
api = wandb.Api(timeout=100)
