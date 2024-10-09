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
    "4rlg66oe",  # Blobs        Accumulate              Bandwidth=SweepConstant
    # "hwdykev5",  # Blobs        Accumulate              Bandwidth=Estimated
    "otef85ps",  # Blobs        Accumulate-Subsample    Bandwidth=SweepConstant
    # "v5e5gk09",  # Blobs        Accumulate-Subsample    Bandwidth=Estimated
    "9qd65w91",  # Blobs        Replace                 Bandwidth=SweepConstant
    # "szvxq5zu",  # Blobs        Replace                 Bandwidth=Estimated
]


runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="rerevisiting-model-collapse-fit-kdes",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username="rylan",
    finished_only=True,
)

# Exclude the non-estimated kernel bandwidths.
runs_configs_df = runs_configs_df[
    ~runs_configs_df["kernel_bandwidth"].isin({"scott", "silverman"})
]
# Convert the remaining kernel bandwidths to floats.
runs_configs_df["kernel_bandwidth"] = runs_configs_df["kernel_bandwidth"].astype(float)

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
        # "tophat": "Top Hat",  # We are no longer using Top Hat kernels b/c inf NLLs b/c lack of support.
    }
)

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

# Keep only a subset for faster and cleaner plotting.
# Take only the model fitting iterations that are multiples of 10.
run_histories_df = run_histories_df[
    (run_histories_df["Model-Fitting Iteration"] % 10 == 0)
    | (run_histories_df["Model-Fitting Iteration"] == 1)
]

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

bandwidth_order = [
    # "scott",
    # "silverman",
    0.01,
    0.0316,
    0.1,
    0.316,
    1.0,
    3.16,
    10.0,
    31.6,
    100.0,
]

for (dataset,), subset_extended_run_histories_df in extended_run_histories_df.groupby(
    ["Dataset"]
):
    # plt.close()
    # g = sns.relplot(
    #     data=subset_extended_run_histories_df,
    #     kind="line",
    #     x=r"Bandwidth $h$",
    #     y="NLL on Real Data (Test)",
    #     col="Setting",
    #     col_order=["Replace", "Accumulate-Subsample", "Accumulate"],
    #     row="Num. Samples per Iteration",
    #     row_order=[10, 32, 100, 316, 1000],
    #     hue="Model-Fitting Iteration",
    #     style="Kernel",
    #     style_order=["Gaussian"],
    #     palette="Spectral_r",
    #     # legend="full",
    #     facet_kws={"sharex": True, "sharey": "row", "margin_titles": True},
    # )
    # g.set(xscale="log", yscale="log")
    # g.set_titles(
    #     col_template="{col_name}",
    #     row_template="Samples per Iteration: {row_name}",
    # )
    # sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    # src.plot.save_plot_with_multiple_extensions(
    #     plot_dir=results_dir,
    #     plot_filename=f"y=nll_x=bandwidth_col=setting_hue=iter_row=samplesperiter_dataset={dataset.lower().replace(' ', '')}",
    # )
    # plt.show()
    # print()

    plt.close()
    g = sns.catplot(
        data=subset_extended_run_histories_df,
        kind="point",
        x=r"Bandwidth $h$",
        y="NLL on Real Data (Test)",
        col="Setting",
        col_order=["Replace", "Accumulate-Subsample", "Accumulate"],
        row="Num. Samples per Iteration",
        row_order=[10, 32, 100, 316, 1000],
        hue="Model-Fitting Iteration",
        palette="Spectral_r",
        margin_titles=True,
        legend="full",
    )
    g.set(yscale="log")
    g.set_titles(
        col_template="{col_name}",
        row_template="Num. Samples per Iter: {row_name}",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    # Add title of the dataset.
    # plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Dataset: {dataset}")
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"y=nll_x=bandwidth_col=setting_hue=iter_row=samplesperiter_dataset={dataset.lower().replace(' ', '')}",
    )
    plt.show()


print("Finished running 11_kde_bandwidth_tests.py")