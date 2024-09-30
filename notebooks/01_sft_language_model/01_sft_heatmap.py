import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import scipy.stats
import seaborn as sns
import wandb
import ast

import src.analyze
import src.plot


# refresh = False
refresh = True
WANDB_PROJ = "heatmap2"

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=refresh,
)

wandb_username = "jkazdan"
sweeps = wandb.Api().project(WANDB_PROJ).sweeps()
sweep_names = [sweep.id for sweep in sweeps]
wandb_sweep_ids = sweep_names


runs_configs_df_1: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path=WANDB_PROJ,
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb_username,
    finished_only=True,
)

# runs_configs_df_2: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
#     wandb_project_path=WANDB_PROJ,
#     data_dir=data_dir,
#     sweep_ids=wandb_sweep_ids[10:],
#     refresh=refresh,
#     wandb_username=wandb_username,
#     finished_only=True,
# )

# for ele in runs_configs_df_1.T.iterrows():
#     print(ele)

runs_configs_df = runs_configs_df_1.iloc[
    5:
]  # pd.concat([runs_configs_df_1, runs_configs_df_2])
# Add the number of model fitting iterations.


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

runs_histories_df_1: pd.DataFrame = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path=WANDB_PROJ,
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb_username,
    wandb_run_history_samples=100000000,  # Make sure we grab _all_ the data.
)

# runs_histories_df_2: pd.DataFrame = src.analyze.download_wandb_project_runs_histories(
#     wandb_project_path=WANDB_PROJ,
#     data_dir=data_dir,
#     sweep_ids=wandb_sweep_ids[20:],
#     refresh=refresh,
#     wandb_username=wandb_username,
#     wandb_run_history_samples=100000000,  # Make sure we grab _all_ the data.
# )

runs_histories_df = (
    runs_histories_df_1  # pd.concat([runs_histories_df_1, runs_histories_df_2])
)

runs_configs_df["Setting"] = "Mixture"
runs_histories_df["Setting"] = "Mixture"
print("########################################")
# runs_configs_df, runs_histories_df = src.analyze.duplicate_real_data_runs(
#     runs_configs_df=runs_configs_df,
#     runs_histories_df=runs_histories_df,
# )

print(runs_configs_df)
plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="line",
    x="num_real",
    y="eval/loss",
    hue="num_synthetic",
    col_order=["Replace", "Accumulate"],
    marker="o",
    markersize=15,
    palette="coolwarm",
)
g.set(xscale="symlog")
g.set_axis_labels(
    x_var="Real data points", y_var="Eval Cross Entropy on Real Data", fontsize=20
)
g.set(xlim=(-0.1, 200000))
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="heatmap_validation_loss",
)
plt.show()

print(runs_configs_df)
plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="line",
    x="num_real",
    y="num_synthetic",
    hue="eval/loss",
    col_order=["Replace", "Accumulate"],
    marker="o",
    markersize=15,
    palette="coolwarm",
)
g.set(xscale="symlog")
g.set(yscale="symlog")
g.set(xlim=(-0.1, 200000), ylim=(-0.1, 200000))
g.set_axis_labels(x_var="Real data points", y_var="Synthetic data points", fontsize=20)
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="heatmap_validation_loss_hue_loss",
)
plt.show()

plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="line",
    x="num_real",
    y="train/loss",
    hue="num_synthetic",
    col_order=["Replace", "Accumulate"],
    marker="o",
    markersize=15,
    palette="coolwarm",
)
g.set_axis_labels(y_var="Train Cross Entropy on Real Data", fontsize=20)
g.set(xlim=(-0.1, 200000))
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="heatmap_train_loss",
)
plt.show()


# extended_run_histories_df = runs_histories_df.merge(
#     runs_configs_df[["run_id", "Model Fitting Iteration"]],
#     left_on="run_id",
#     right_on="run_id",
# )


# plt.close()
# g = sns.relplot(
#     data=extended_run_histories_df,
#     kind="line",
#     x="train/epoch",
#     y="eval/loss",
#     col="Setting",
#     hue="Model Fitting Iteration",
# )
# g.set_yticklabels(fontsize=10)
# g.set_axis_labels("Epoch", "Eval Cross Entropy on Real Data")
# g.set_titles("{col_name}")
# sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="y=eval_loss_x=epoch_col=setting_hue=model_fitting_iteration",
# )
# # plt.show()

# # Visualize each individual run's learning curve.
# runs_learning_curves_dir = os.path.join(results_dir, "learning_curves_per_run")
# os.makedirs(runs_learning_curves_dir, exist_ok=True)
# for run_id, run_history_df in extended_run_histories_df.groupby("run_id"):
#     # extended_run_histories_df.loc[
#     #     run_history_df.index, "eval/loss_smoothed"
#     # ] = run_history_df["eval/loss"].rolling(window=10).mean()
#     plt.close()
#     sns.lineplot(
#         data=run_history_df,
#         x="train/epoch",
#         y="eval/loss",
#     )
#     plt.title(f"Run ID: {run_id}")
#     src.plot.save_plot_with_multiple_extensions(
#         plot_dir=runs_learning_curves_dir,
#         plot_filename=f"y=eval_loss_x=epoch_run_id={run_id}",
#     )
#     # plt.show()


print("Finished running notebooks/01_sft_language_model.py")
