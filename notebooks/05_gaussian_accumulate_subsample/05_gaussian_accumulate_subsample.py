import ast
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


refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


sweep_ids = [
    "k3mipjqi",  # Gaussian fitting experiment (~5k runs); Accumulate and Replace
    "2ssx4zho",  # Gaussian fitting experiment (~2.5k runs); Accumulate-Subsample.
]

run_histories_df: pd.DataFrame = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="rerevisiting-model-collapse-fit-gaussians",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
)

run_histories_df["Task"] = "Gaussian Modeling"

for (data_dim,), run_histories_data_dim_df in run_histories_df.groupby(
    ["Data Dimension"]
):
    plt.close()
    g = sns.relplot(
        data=run_histories_data_dim_df,
        x="Model-Fitting Iteration",
        y="Squared Error of Fit Mean (Numerical)",
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
        "Model-Fitting Iteration",
        r"$\lvert \lvert \hat{\mu}_n - \mu_0 \lvert \lvert_2^2$",
    )
    g.set_titles(
        col_template="{col_name}",
        row_template="{row_name}",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"squared_error_of_fit_mean_vs_model_fitting_iteration_by_noise_col=setting_dim={data_dim}",
    )
    # plt.show()

    plt.close()
    g = sns.relplot(
        data=run_histories_data_dim_df,
        kind="line",
        x="Model-Fitting Iteration",
        y="Det of Fit Cov / Det of Init Cov (Numerical)",
        hue="Num. Samples per Iteration",
        hue_norm=matplotlib.colors.LogNorm(),
        col="Setting",
        col_order=["Replace", "Accumulate-Subsample", "Accumulate"],
        row="Task",
        facet_kws={"sharey": True, "sharex": True, "margin_titles": True},
        palette="cool",
        # palette="mako_r",
        legend="full",
    )
    g.set_axis_labels(
        "Model-Fitting Iteration", r"$det(\hat{\Sigma}_n) / det(\Sigma_0)$"
    )
    g.set_titles(
        col_template="{col_name}",
        row_template="{row_name}",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"determinant_of_fit_cov_vs_model_fitting_iteration_by_noise_col=setting_dim={data_dim}",
    )
    # plt.show()

    plt.close()
    g = sns.relplot(
        data=run_histories_data_dim_df,
        kind="line",
        x="Model-Fitting Iteration",
        y="Trace of Fit Cov / Trace of Init Cov (Numerical)",
        hue="Num. Samples per Iteration",
        hue_norm=matplotlib.colors.LogNorm(),
        col="Setting",
        col_order=["Replace", "Accumulate-Subsample", "Accumulate"],
        row="Task",
        facet_kws={"sharey": True, "sharex": True, "margin_titles": True},
        palette="cool",
        # palette="mako_r",
        legend="full",
    )
    g.set_axis_labels("Model-Fitting Iteration", r"$Tr(\hat{\Sigma}_n) / Tr(\Sigma_0)$")
    g.set_titles(
        col_template="{col_name}",
        row_template="{row_name}",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"trace_of_fit_cov_vs_model_fitting_iteration_by_noise_col=setting_dim={data_dim}",
    )
    # plt.show()


print("Finished running 05_gaussian_accumulate_subsample.py")
