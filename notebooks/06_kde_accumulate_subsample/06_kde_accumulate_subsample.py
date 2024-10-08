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
    "2crqw2ne",  # Blobs (~3k runs); Accumulate and Replace.
    "n1sl4eew",  # Blobs (~1.5k runs); Accumulate-Subsample.
    "r66vkvsf",  # Circles (~3k runs); Accumulate and Replace.
    "tlstv5l3",  # Circles (~1.5k runs); Accumulate-Subsample.
    "tq2fnp98",  # Moons (~3k runs); Accumulate and Replace.
    "3w2og2ru",  # Moons (~1.5k runs); Accumulate-Subsample.
    "hutjomj9",  # Swiss Roll (~3k runs); Accumulate and Replace.
    "cffq8fyu",  # Swiss Roll (~1.5k runs); Accumulate-Subsample.
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

# Remove kernel Top Hat because most of the negative log likelihoods are inf.
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

# # Only plot "Dataset" == "Swiss Roll" for now.
# runs_configs_df = runs_configs_df[runs_configs_df["Dataset"] == "Swiss Roll"]


run_histories_df: pd.DataFrame = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="rerevisiting-model-collapse-fit-kdes",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
)

run_histories_df["Task"] = "Kernel Density Estimation"

run_histories_df = run_histories_df.rename(
    columns={"Mean Negative Log Prob (Test)": "NLL on Real Data (Test)"}
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

for (dataset,), dataset_extended_run_histories_df in extended_run_histories_df.groupby(
    ["Dataset"]
):
    plt.close()
    g = sns.relplot(
        data=dataset_extended_run_histories_df,
        kind="line",
        x="Model-Fitting Iteration",
        y="NLL on Real Data (Test)",
        col="Setting",
        col_order=["Replace", "Accumulate-Subsample", "Accumulate"],
        row="Task",
        hue="Num. Samples per Iteration",
        hue_norm=matplotlib.colors.LogNorm(),
        hue_order=[10, 32, 100, 316, 1000],
        style="Kernel",
        style_order=["Gaussian"],
        palette="cool",
        # palette="mako_r",
        legend="full",
        facet_kws={"sharex": True, "sharey": "row", "margin_titles": True},
    )
    g.set(yscale="log")
    g.set_titles(
        col_template="{col_name}",
        row_template="{row_name}",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"neg_log_likelihood_vs_model_fitting_iteration_by_noise_col=setting_dataset={dataset.lower().replace(' ', '')}",
    )
    # plt.show()

for (
    dataset,
    bandwidth,
), dataset_bandwidth_group_df in extended_run_histories_df.groupby(
    ["Dataset", r"Bandwidth $h$"]
):
    plt.close()
    g = sns.relplot(
        data=dataset_bandwidth_group_df,
        kind="line",
        x="Model-Fitting Iteration",
        y="NLL on Real Data (Test)",
        col="Setting",
        col_order=["Replace", "Accumulate-Subsample", "Accumulate"],
        row="Task",
        hue="Num. Samples per Iteration",
        hue_norm=matplotlib.colors.LogNorm(),
        hue_order=[10, 32, 100, 316, 1000],
        style="Kernel",
        style_order=["Gaussian"],
        palette="cool",
        # palette="mako_r",
        legend="full",
        facet_kws={"sharex": True, "sharey": "row", "margin_titles": True},
    )
    g.set(yscale="log")
    g.set_titles(
        col_template="{col_name}",
        row_template="{row_name}",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"neg_log_likelihood_vs_model_fitting_iteration_by_noise_col=setting_dataset={dataset.lower().replace(' ', '')}_bandwidth={bandwidth}",
    )
    # plt.show()

print("Finished running 06_kde_accumulate_subsample.py")
