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
    "k3mipjqi",  # Gaussian fitting experiment (~5k runs).
]

run_histories_df: pd.DataFrame = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="rerevisiting-model-collapse-fit-gaussians",
    data_dir=data_dir,
    sweep_ids=sweep_ids,
    refresh=refresh,
    wandb_username=wandb.api.default_entity,
)

# Plot the covariances over time as Gaussian PDFs.
# Define the range for the x-axis (support)
x = np.linspace(-4, 4, 1000)
model_fitting_iteration_indices = sorted(
    run_histories_df["Model-Fitting Iteration"].unique()
)
cmap = matplotlib.cm.get_cmap("Spectral_r", len(model_fitting_iteration_indices))

for num_samples_per_iter in run_histories_df["Num. Samples per Iteration"].unique():
    plt.close()
    fig, axes = plt.subplots(
        figsize=(12, 4),
        ncols=2,
        nrows=1,
        sharey=True,
        sharex=True,
    )
    for ax_idx, setting in enumerate(["Replace", "Accumulate"]):
        ax = axes[ax_idx]
        for iteration_idx, iteration_group in run_histories_df.groupby(
            "Model-Fitting Iteration"
        ):
            iteration_group = iteration_group[
                (iteration_group["Setting"] == setting)
                & (
                    iteration_group["Num. Samples per Iteration"]
                    == num_samples_per_iter
                )
                & (iteration_group["Data Dimension"] == 1)
            ]
            cov = iteration_group["Fit Covariance (Numerical)"].mean()
            y = scipy.stats.norm.pdf(x, 0, cov)
            ax.plot(x, y, color=cmap(iteration_idx))
        ax.set_title(f"{setting}")
        ax.set(xlim=(-3.0, 3.0))
        # ax.set_xlabel("Mean")
        if ax_idx == 0:
            ax.set_ylabel("Density")

    # Create a ScalarMappable and colorbar
    norm = matplotlib.colors.Normalize(
        vmin=min(model_fitting_iteration_indices),
        vmax=max(model_fitting_iteration_indices),
    )
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib
    plt.tight_layout()
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="vertical", pad=0.02)
    cbar.set_label("Model-Fitting Iteration")
    # Need to do this manually because of the tight layout.
    for extension in {"png", "pdf"}:
        plt.savefig(
            f"{results_dir}/fit_covariance_pdf_by_model_fitting_iteration_samples={int(num_samples_per_iter)}.{extension}",
            bbox_inches="tight",
            dpi=300,
        )
    # plt.show()


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
        col_order=["Replace", "Accumulate"],
        kind="line",
        facet_kws={"sharey": True, "sharex": True, "margin_titles": True},
        palette="mako_r",
        legend="full",
    )
    g.set(yscale="log")
    g.set_axis_labels(
        "Model-Fitting Iteration",
        r"$\lvert \lvert \hat{\mu}_n - \mu_0 \lvert \lvert_2^2$",
    )
    g.set_titles(
        col_template="{col_name}",
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
        x="Model-Fitting Iteration",
        y="Det of Fit Cov / Det of Init Cov (Numerical)",
        hue="Num. Samples per Iteration",
        hue_norm=matplotlib.colors.LogNorm(),
        col="Setting",
        col_order=["Replace", "Accumulate"],
        kind="line",
        facet_kws={"sharey": True, "sharex": True, "margin_titles": True},
        palette="mako_r",
        legend="full",
    )
    g.set_axis_labels(
        "Model-Fitting Iteration", r"$det(\hat{\Sigma}_n) / det(\Sigma_0)$"
    )
    g.set_titles(
        col_template="{col_name}",
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
        x="Model-Fitting Iteration",
        y="Trace of Fit Cov / Trace of Init Cov (Numerical)",
        hue="Num. Samples per Iteration",
        hue_norm=matplotlib.colors.LogNorm(),
        col="Setting",
        col_order=["Replace", "Accumulate"],
        kind="line",
        facet_kws={"sharey": True, "sharex": True, "margin_titles": True},
        palette="mako_r",
        legend="full",
    )
    g.set_axis_labels("Model-Fitting Iteration", r"$Tr(\hat{\Sigma}_n) / Tr(\Sigma_0)$")
    g.set_titles(
        col_template="{col_name}",
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=results_dir,
        plot_filename=f"trace_of_fit_cov_vs_model_fitting_iteration_by_noise_col=setting_dim={data_dim}",
    )
    # plt.show()


print("Finished running 00_gaussian_fitting.py")
