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
    "l7gatv8d",  # Linear Regression        Accumulate              Dim=1
    "a163ryjz",  # Linear Regression        Accumulate              Dim=3
    "25ter2ya",  # Linear Regression        Accumulate              Dim=10
    "pjywsfx0",  # Linear Regression        Accumulate              Dim=32
    "dy6ghxo1",  # Linear Regression        Accumulate              Dim=100
    "nqc8d6d1",  # Linear Regression        Accumulate-Subsample    Dim=1
    "01yterpm",  # Linear Regression        Accumulate-Subsample    Dim=3
    "x444d2q6",  # Linear Regression        Accumulate-Subsample    Dim=10
    "j27s0v5x",  # Linear Regression        Accumulate-Subsample    Dim=32
    "vr0x9v2w",  # Linear Regression        Accumulate-Subsample    Dim=100
    "zv4s906m",  # Linear Regression        Replace                 Dim=1
    "eqxf6svs",  # Linear Regression        Replace                 Dim=3
    "trmmfbt4",  # Linear Regression        Replace                 Dim=10
    "09flzov3",  # Linear Regression        Replace                 Dim=32
    "it5d7fh7",  # Linear Regression        Replace                 Dim=100
]

runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="rerevisiting-model-collapse-fit-lin-regr",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username="rylan",
    finished_only=True,
)

# Rename columns to make them a little nicer.
runs_configs_df = runs_configs_df.rename(
    columns={
        "data_dim": "Data Dimension",
        "sigma_squared": r"$\sigma^2$",
        "num_samples_per_iteration": "Num. Samples per Iteration",
        "setting": "Setting",
    }
)


run_histories_df: pd.DataFrame = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="rerevisiting-model-collapse-fit-lin-regr",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
)

run_histories_df["Task"] = "Linear Regression"

extended_run_histories_df = run_histories_df.merge(
    runs_configs_df[
        [
            "run_id",
            "Data Dimension",
            r"$\sigma^2$",
            "Num. Samples per Iteration",
            "Setting",
        ]
    ],
    on="run_id",
    how="inner",
)


for (
    data_dim,
    sigma_squared,
), extended_run_histories_subset_df in extended_run_histories_df.groupby(
    ["Data Dimension", r"$\sigma^2$"]
):
    plt.close()
    g = sns.relplot(
        data=extended_run_histories_subset_df,
        x="Model-Fitting Iteration",
        y="Mean Squared Error (Test)",
        hue="Num. Samples per Iteration",
        hue_norm=matplotlib.colors.LogNorm(),
        col="Setting",
        col_order=["Replace", "Accumulate-Subsample", "Accumulate"],
        row="Task",
        kind="line",
        facet_kws={"sharey": True, "sharex": True, "margin_titles": True},
        palette="cool",
        legend="full",
    )
    g.set(yscale="log")
    g.set_axis_labels(
        x_var="Model-Fitting Iteration",
        # y_var=r"$\mathbb{E}[\lvert \lvert \hat{y}^{(t)} - y \lvert \lvert_2^2 ]$",
    )
    g.set_titles(
        col_template="{col_name}",
        row_template="{row_name}",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"squared_error_of_fit_mean_vs_model_fitting_iteration_by_noise_col=setting_dim={data_dim}_sigmasq={sigma_squared}",
    )
    # plt.show()


print("Finished running 07_lin_regr_accumulate_subsample.py")
