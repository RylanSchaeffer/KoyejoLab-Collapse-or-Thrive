import matplotlib.pyplot as plt
import matplotlib.transforms
import os
import pandas as pd
import seaborn as sns
import wandb

import src.analyze
import src.plot


refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


wandb_sweep_ids = [
    "2crqw2ne",  # Blobs (~6k runs).
    "r66vkvsf",  # Circles (~6k runs).
    "tq2fnp98",  # Moons (~6k runs).
    "hutjomj9",  # Swiss Roll (~6k runs).
]

runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="rerevisiting-model-collapse-fit-kdes",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username="rylan",
    finished_only=True,
)

keys_to_extract_from_cols = [
    (
        "data_config",
        "dataset_name",
        "Dataset",
    ),
]

# Extract what we need from nested dictionaries.
for key_to_extract in keys_to_extract_from_cols:
    runs_configs_df = src.analyze.extract_key_value_from_df_col(
        df=runs_configs_df,
        col_name=key_to_extract[0],
        key_in_dict=key_to_extract[1],
        new_col_name=key_to_extract[2],
    )


# Rename columns to make them a little nicer.
runs_configs_df = runs_configs_df.rename(
    columns={
        "kernel": "Kernel",
        "kernel_bandwidth": r"Bandwidth $h$",
        "num_samples_per_iteration": "Num. Samples per Iteration",
        "setting": "Setting",
    }
)

# Rename "Kernel" column values.
runs_configs_df["Kernel"] = runs_configs_df["Kernel"].map(
    {
        "gaussian": "Gaussian",
        "tophat": "Top Hat",
    }
)

# Remove kernel Top Hat because most of the negative log likelihoods are NaN.
runs_configs_df = runs_configs_df[runs_configs_df["Kernel"] == "Gaussian"]

# Rename "Dataset" column values.
runs_configs_df["Dataset"] = runs_configs_df["Dataset"].map(
    {
        "blobs": "Blobs",
        "circles": "Circles",
        "moons": "Moons",
        "swiss_roll": "Swiss Roll",
    }
)


run_histories_df: pd.DataFrame = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="rerevisiting-model-collapse-fit-kdes",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
)

run_histories_df = run_histories_df.rename(
    columns={"Mean Negative Log Prob (Test)": "Eval NLL on Real Data"}
)

extended_run_histories_df = run_histories_df.merge(
    runs_configs_df[
        [
            "run_id",
            "Dataset",
            "Kernel",
            r"Bandwidth $h$",
            "Num. Samples per Iteration",
            "Setting",
        ]
    ],
    on="run_id",
    how="inner",
)

plt.close()
g = sns.relplot(
    data=extended_run_histories_df,
    kind="line",
    x="Model-Fitting Iteration",
    y="Eval NLL on Real Data",
    col="Setting",
    col_order=["Replace", "Accumulate"],
    row="Dataset",
    row_order=["Blobs", "Circles", "Moons", "Swiss Roll"],
    hue="Num. Samples per Iteration",
    hue_norm=matplotlib.colors.LogNorm(),
    style="Kernel",
    style_order=["Gaussian", "Top Hat"],
    palette="cool",
    # palette="mako_r",
    legend="full",
    facet_kws={"sharex": True, "sharey": "row", "margin_titles": True},
)
g.set(yscale="log")
g.set_titles(
    col_template="{col_name}",
    row_template="Dataset: {row_name}",
)
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename=f"neg_log_likelihood_vs_model_fitting_iteration_by_noise_col=setting_row=dataset",
)
# plt.show()

for bandwidth, bandwidth_group_df in extended_run_histories_df.groupby(
    r"Bandwidth $h$"
):
    plt.close()
    g = sns.relplot(
        data=bandwidth_group_df,
        kind="line",
        x="Model-Fitting Iteration",
        y="Eval NLL on Real Data",
        col="Setting",
        col_order=["Replace", "Accumulate"],
        row="Dataset",
        row_order=["Blobs", "Circles", "Moons", "Swiss Roll"],
        hue="Num. Samples per Iteration",
        hue_norm=matplotlib.colors.LogNorm(),
        style="Kernel",
        style_order=["Gaussian", "Top Hat"],
        palette="cool",
        # palette="mako_r",
        legend="full",
        facet_kws={"sharex": True, "sharey": "row", "margin_titles": True},
    )
    g.set(yscale="log")
    g.set_titles(
        col_template="{col_name}",
        row_template="Dataset: {row_name}",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"neg_log_likelihood_vs_model_fitting_iteration_by_noise_col=setting_row=dataset_bandwidth={bandwidth}",
    )
    # plt.show()

print("Finished running 02_kde_fitting.py")
