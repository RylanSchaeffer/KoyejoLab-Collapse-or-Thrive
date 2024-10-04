import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LogNorm, SymLogNorm
from sklearn.metrics import r2_score

import src.analyze
import src.plot


refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# wandb_username = "jkazdan"
# wandb_sweep_ids = [
#     "pqcd6apc",
#     "jj3km0np",
#     "7s2ut03h",
# ]
# wandb_project_path = "heatmap3"

wandb_username = "rylan"
wandb_sweep_ids = ["nbiswnvi"]
wandb_project_path = "model-collapse-value-synthetic"

runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path=wandb_project_path,
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
    wandb_project_path=wandb_project_path,
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb_username,
    wandb_run_history_samples=100000000,  # Make sure we grab _all_ the data.
)

extended_runs_histories_df = runs_histories_df.merge(
    runs_configs_df[["run_id", "num_real", "num_synthetic", "seed", "dataset"]],
    on="run_id",
    how="inner",
)


runs_configs_df["Setting"] = "Mixture"
runs_histories_df["Setting"] = "Mixture"

plt.close()
g = sns.relplot(
    data=runs_configs_df,
    # data=runs_configs_df[runs_configs_df["num_real"] > 0],
    kind="line",
    x="num_real",
    y="eval/loss",
    hue="num_synthetic",
    hue_norm=LogNorm(),
    col="num_synthetic",
    col_wrap=4,
    marker="o",
    markersize=7,
    # palette="coolwarm",
    palette="Spectral_r",
    legend="full",
)
g.set(xlim=(2e2, 1.5e4), xscale="symlog", yscale="log")
g.set_axis_labels(
    x_var="Num. Real Data", y_var="Cross Entropy on Real Data (Test)", fontsize=20
)
g.set_titles(col_template="{col_name} Synthetic Data")
legend = g.legend
legend.set_title("Num. Synthetic Data")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=cross_entropy_x=num_real_hue=num_synthetic_col=num_synthetic",
)
plt.show()


plt.close()
g = sns.relplot(
    data=runs_configs_df,
    # data=runs_configs_df[runs_configs_df["num_real"] > 0],
    kind="line",
    x="num_synthetic",
    y="eval/loss",
    hue="num_real",
    hue_norm=LogNorm(),
    col="num_real",
    col_wrap=4,
    marker="o",
    markersize=15,
    # palette="coolwarm",
    palette="Spectral_r",
    legend="full",
)
g.set(xlim=(2e2, 1.5e4), xscale="symlog", yscale="log")
g.set_axis_labels(
    x_var="Num. Synthetic Data", y_var="Cross Entropy on Real Data (Test)", fontsize=20
)
g.set_titles(col_template="{col_name} Real Data")
legend = g.legend
legend.set_title("Num. Real Data")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=cross_entropy_x=num_synthetic_hue=num_real_col=num_real",
)
plt.show()


# loss v. number of real datapoints
plt.close()
g = sns.relplot(
    data=runs_configs_df[runs_configs_df["num_real"] > 0],
    kind="line",
    x="num_real",
    y="eval/loss",
    hue="num_synthetic",
    # hue_norm=SymLogNorm(linthresh=1.0),
    col_order=["Replace", "Accumulate"],
    marker="o",
    markersize=7,
    # palette="coolwarm",
    palette="Spectral_r",
    legend="full",
)
# # Add dashed horizontal lines of the loss for each "num_synthetic" where "num_real" = 0.
# # Extract the color palette
# palette = sns.color_palette(
#     "coolwarm", n_colors=len(runs_configs_df["num_synthetic"].unique())
# )
# color_dict = dict(zip(sorted(runs_configs_df["num_synthetic"].unique()), palette))
# for num_synthetic in runs_configs_df["num_synthetic"].unique():
#     try:
#         loss = runs_configs_df[
#             (runs_configs_df["num_real"] == 0)
#             & (runs_configs_df["num_synthetic"] == num_synthetic)
#         ]["eval/loss"].values[0]
#     except IndexError:
#         # TODO: Remove this once sweeps are done running.
#         continue
#     plt.axhline(
#         y=loss,
#         linestyle="--",
#         color=color_dict[num_synthetic],  # Use the corresponding color
#         zorder=1,
#     )
g.set(xlim=(2e2, 1.5e4), xscale="symlog", yscale="log")
g.set_axis_labels(
    x_var="Real Data Points", y_var="Cross Entropy on Real Data (Test)", fontsize=20
)
legend = g.legend
legend.set_title("Num. Synthetic Data")
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

# X = runs_configs_df["proportion"]
# model = sm.OLS(y, X).fit()
# print(f"The R^2 value for the absolute number of real on the loss is {model.rsquared}")

runs_configs_df["num_data"] = (
    runs_configs_df["num_real"] + runs_configs_df["num_synthetic"]
)

# # eval loss v proportion of real data
# runs_configs_over_seed_df = (
#     runs_configs_df[["eval/loss", "num_data", "proportion", "seed"]]
#     .groupby(
#         [
#             "proportion",
#             "num_data",
#         ]
#     )
#     .mean()
#     .reset_index()
# )

runs_configs_df = runs_configs_df.rename(
    columns={
        "proportion": "Real Data / (Real Data + Synthetic Data)",
        "num_data": "Total Num. Data",
        "num_synthetic": "Num. Synthetic Data",
        "num_real": "Num. Real Data",
    }
)


plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="scatter",
    x="Real Data / (Real Data + Synthetic Data)",
    y="eval/loss",
    hue="Total Num. Data",
    hue_norm=LogNorm(),
    marker="o",
    # palette="coolwarm",
    palette="copper",
)
g.set_axis_labels(y_var="Cross Entropy on Real Data (Test)", fontsize=20)
g.set(
    xscale="log",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="heatmap_proportion_loss",
)
plt.show()


plt.close()
g = sns.relplot(
    data=runs_configs_df[runs_configs_df["Num. Real Data"] > 0],
    kind="line",
    x="Real Data / (Real Data + Synthetic Data)",
    y="eval/loss",
    # style="Total Num. Data",
    hue_norm=SymLogNorm(linthresh=1.0),
    hue="Num. Synthetic Data",
    palette="copper",
)
g.set_axis_labels(y_var="Cross Entropy on Real Data (Test)", fontsize=20)
g.set(
    xscale="log",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=cross_entropy_x=proportion_real_hue=num_synthetic",
)
plt.show()


plt.close()
g = sns.relplot(
    data=runs_configs_df[runs_configs_df["Num. Real Data"] > 0],
    kind="line",
    x="Real Data / (Real Data + Synthetic Data)",
    y="eval/loss",
    # style="Total Num. Data",
    hue_norm=SymLogNorm(linthresh=1.0),
    hue="Num. Real Data",
    palette="copper",
)
g.set_axis_labels(y_var="Cross Entropy on Real Data (Test)", fontsize=20)
g.set(
    xscale="log",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=cross_entropy_x=proportion_real_hue=num_real",
)
plt.show()


plt.close()
g = sns.relplot(
    data=extended_runs_histories_df,
    kind="line",
    x="_step",
    y="eval/loss",
    hue="num_real",
    hue_norm=SymLogNorm(linthresh=1.0),
    style="num_synthetic",
)
plt.show()


print("Finished running notebooks/10_sft_cardinality_vs_proportionality.py")
