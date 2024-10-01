import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import scipy.stats
import seaborn as sns
import wandb
import ast
import statsmodels.api as sm
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score

import src.analyze
import src.plot


refresh = False
# refresh = True

WANDB_PROJ = "heatmap3"

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

wandb_username = "jkazdan"
# sweeps = wandb.Api().project(WANDB_PROJ).sweeps()
wandb_sweep_ids = ["pqcd6apc"]  # [sweep.id for sweep in sweeps]


runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path=WANDB_PROJ,
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb_username,
    finished_only=True,
)

# After this, we now have a column called "dataset" in runs_configs_df.
runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=runs_configs_df,
    col_name="data_config",
    key_in_dict="dataset",
    new_col_name="dataset",
)
runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=runs_configs_df,
    col_name="data_config",
    key_in_dict="num_real",
    new_col_name="num_real",
)

runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=runs_configs_df,
    col_name="data_config",
    key_in_dict="num_synthetic",
    new_col_name="num_synthetic",
)

runs_histories_df: pd.DataFrame = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path=WANDB_PROJ,
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb_username,
    wandb_run_history_samples=100000000,  # Make sure we grab _all_ the data.
)


runs_configs_df["Setting"] = "Mixture"
runs_histories_df["Setting"] = "Mixture"


# loss v. number of real datapoints
print(runs_configs_df)
plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="line",
    x="num_real",
    y="eval/loss",
    hue="num_synthetic",
    hue_norm=LogNorm(),
    col_order=["Replace", "Accumulate"],
    marker="o",
    markersize=7,
    palette="coolwarm",
)
g.set(xscale="symlog", yscale="log")
g.set_axis_labels(
    x_var="Real Data Points", y_var="Eval Cross Entropy on Real Data", fontsize=20
)
g.set(xlim=(-1, 3e4))
legend = g.legend
legend.set_title("Num. Synthetic Datapoints")

sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="heatmap_validation_loss",
)
plt.show()

# num synthetic v. number of real datapoints
plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="line",
    x="num_real",
    y="num_synthetic",
    hue="eval/loss",
    col_order=["Replace", "Accumulate"],
    marker="o",
    markersize=7,
    palette="coolwarm",
)
g.set(xscale="symlog")
g.set(yscale="symlog")
g.set(xlim=(100, 3e4), ylim=(100, 3e4))
g.set_axis_labels(x_var="Real Data Points", y_var="Synthetic Data Points", fontsize=20)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="heatmap_validation_loss_hue_loss",
)
plt.show()

# train loss v. num synthetic
plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="line",
    x="num_real",
    y="train/loss",
    hue="num_synthetic",
    col_order=["Replace", "Accumulate"],
    marker="o",
    markersize=7,
    palette="coolwarm",
)
g.set_axis_labels(y_var="Train Cross Entropy on Real Data", fontsize=20)
g.set(xlim=(-1, 3e4))
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="heatmap_train_loss",
)
plt.show()

# compute the proportion of real data
runs_configs_df["proportion"] = runs_configs_df["num_real"] / (
    runs_configs_df["num_synthetic"] + runs_configs_df["num_real"]
)

#
X = runs_configs_df["num_real"]
y = runs_configs_df["eval/loss"]
model = sm.OLS(y, X).fit()

print(f"The R^2 value for the proportion of real on the loss is {model.rsquared}")

X = runs_configs_df["proportion"]
model = sm.OLS(y, X).fit()
print(f"The R^2 value for the absolute number of real on the loss is {model.rsquared}")

runs_configs_df["n_data"] = (
    runs_configs_df["num_real"] + runs_configs_df["num_synthetic"]
)

# eval loss v proportion of real data
plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="scatter",
    x="proportion",
    y="eval/loss",
    hue="n_data",
    col_order=["Replace", "Accumulate"],
    marker="o",
    palette="coolwarm",
)
g.set_axis_labels(y_var="Train Cross Entropy on Real Data", fontsize=20)
g.set(xlim=(0, 1))
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="heatmap_proportion_loss",
)
plt.show()

print("Finished running notebooks/01_sft_language_model.py")
