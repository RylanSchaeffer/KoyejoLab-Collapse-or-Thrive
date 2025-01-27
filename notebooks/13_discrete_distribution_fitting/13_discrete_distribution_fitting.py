import ast
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import os
import pandas as pd
import scipy.stats
import seaborn as sns
import wandb

import src.analyze
import src.plot


# refresh = False
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


sweep_ids = [
    "fjey4wtd",  # Replace. Uniform initialization. Part 1.
    "fiqglgv9",  # Replace. Uniform initialization. Part 2.
]

# run_histories_df: pd.DataFrame = src.analyze.download_discrete_distribution_fitting_run_histories(
#     wandb_project_path="rerevisiting-model-collapse-fit-discrete-distributions",
#     data_dir=data_dir,
#     sweep_ids=sweep_ids,
#     refresh=refresh,
#     wandb_username=wandb.api.default_entity,
# )

run_histories_df: pd.DataFrame = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="rerevisiting-model-collapse-fit-discrete-distributions",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
)

# Compute the number of steps before total collapse.
num_steps_before_total_collapse_df = (
    run_histories_df.groupby(["run_id"])["Model-Fitting Iteration"].max().reset_index()
)
num_steps_before_total_collapse_df = num_steps_before_total_collapse_df.merge(
    run_histories_df,
    on=["run_id", "Model-Fitting Iteration"],
    how="left",
)

avg_num_steps_before_total_collapse_df = (
    num_steps_before_total_collapse_df.groupby(
        ["Num. Samples per Iteration", "Num. Outcomes"]
    )["Model-Fitting Iteration"]
    .mean()
    .reset_index()
)
# Plot the number of steps before total collapse.
plt.close()
# plt.figure(figsize=(12, 8))
plt.figure(figsize=(14, 10))
g = sns.scatterplot(
    data=avg_num_steps_before_total_collapse_df,
    x="Num. Samples per Iteration",
    y="Model-Fitting Iteration",
    hue="Num. Samples per Iteration",
    hue_norm=matplotlib.colors.LogNorm(),
    style="Num. Outcomes",
    palette="cool",
    s=500,
)
g.set(
    yscale="log",
    xscale="log",
    ylabel="Expected Number of Model-Fitting Iterations to Total Collapse",
)
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=model_fitting_iteration_x=num_samples_per_iteration_hue=num_outcomes",
)
plt.show()

# TODO: Cache this replication.
run_histories_df = src.analyze.replicate_final_rows_up_to_group_max(
    df=run_histories_df,
    group_cols=("Num. Samples per Iteration", "Num. Outcomes"),
    iteration_col="Model-Fitting Iteration",
    run_id_col="run_id",
)

run_histories_df["Fraction of Initial Entropy"] = (
    run_histories_df["Entropy"] / run_histories_df["Initial Entropy"]
)

# Plot the entropy over model-fitting iterations.
plt.close()
# plt.figure(figsize=(12, 8))
plt.figure(figsize=(14, 10))
g = sns.lineplot(
    data=run_histories_df,
    x="Model-Fitting Iteration",
    y="Fraction of Initial Entropy",
    hue="Num. Samples per Iteration",
    hue_norm=matplotlib.colors.LogNorm(),
    style="Num. Outcomes",
    palette="cool",
)
g.set(
    xscale="symlog",
    xlim=(0, None),
    ylim=(0.0, 1.0),
    ylabel="Fraction of Initial Distribution's Entropy",
)
# sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=fraction_of_initial_entropy_x=model_fitting_iteration_hue=num_samples_per_iteration_style=num_outcomes",
)
plt.show()


print("Finished notebooks/13_discrete_distribution_fitting!")
