import matplotlib.colors
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

import src.analyze
import src.plot


# refresh = False
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=refresh,
)

wandb_username = "harvardparkesateams"
wandb_run_ids = [
    "rj0agnxk",
    "3eqk3tcq",
    "984nwhql",
    "jfrxq8h4",
    "886ef2n9",
    "mmaj1xyc",
    "dkayk5me",
    "5fq9g749",
    "67zs2v8c",
    "cdhf99ml",
    "kht4g1qx",
    "13l66wpw",
]

runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="ft_collapse",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb_username,
    finished_only=True,
)
