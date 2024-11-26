import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import seaborn as sns

import src.analyze
import src.plot


# refresh = False
refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=refresh,
)

wandb_username = "rylan"
wandb_sweep_ids = [
    "tb6c1gtr",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate             Iteration1
    "ct9m8x0l",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate             Iteration2
    "p7hjia80",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate             Iteration3
    "akm93fto",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate             Iteration4
    "v8kta96l",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate             Iteration5
    "bygiyqhg",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate             Iteration6
    "ds1ukfox",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate             Iteration7
    "vwu954ge",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate             Iteration8
    "7w4kg2jm",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate             Iteration9
    "8rfj4fko",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate             Iteration10
    "n2ren5e9",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration1
    "5i6bj5re",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration2
    "n8t9j5ee",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration3
    "rk8y6anj",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration4
    "giodjp03",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration5
    "ugblwb61",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration6
    "qias2rm1",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration7
    "70nfygoq",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration8
    "3fd19zjs",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration9
    "hnk0v7gf",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration10
    "cybhcd41",  # HelpSteer2   Gemma2-9B   Paradigm=Accumulate             Iteration1
    "oy29fswj",  # HelpSteer2   Gemma2-9B   Paradigm=Accumulate             Iteration2
    "3ueeirbr",  # HelpSteer2   Gemma2-9B   Paradigm=Accumulate             Iteration3
    "ge1iow70",  # HelpSteer2   Gemma2-9B   Paradigm=Accumulate             Iteration4
    "sqmqrzvq",  # HelpSteer2   Gemma2-9B   Paradigm=Accumulate             Iteration5
    "q6rvic5l",  # HelpSteer2   Gemma2-9B   Paradigm=Replace                Iteration1
    "qg8stube",  # HelpSteer2   Gemma2-9B   Paradigm=Replace                Iteration2
    "9r3j5dg5",  # HelpSteer2   Gemma2-9B   Paradigm=Replace                Iteration3
    "1p9u2eq9",  # HelpSteer2   Gemma2-9B   Paradigm=Replace                Iteration4
    "nyeq61cy",  # HelpSteer2   Gemma2-9B   Paradigm=Replace                Iteration5
    "uaoh66ds",  # HelpSteer2   Gemma2-9B   Paradigm=Replace                Iteration6
    "2touoqm7",  # HelpSteer2   Gemma2-27B  Paradigm=Accumulate             Iteration1
    "jrhsp696",  # HelpSteer2   Gemma2-27B  Paradigm=Accumulate             Iteration2
    "1gbt31up",  # HelpSteer2   Gemma2-27B  Paradigm=Accumulate             Iteration3
    "x3ql72ir",  # HelpSteer2   Gemma2-27B  Paradigm=Replace                Iteration1
    "ktkk7fet",  # HelpSteer2   Gemma2-27B  Paradigm=Replace                Iteration2
    "udchhhjf",  # HelpSteer2   Gemma2-27B  Paradigm=Replace                Iteration3
]

runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="ft_collapse",
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
    col_name="model_config",
    key_in_dict="initial_model_name_or_path",
    new_col_name="Model",
)

runs_configs_df["Model"] = runs_configs_df["Model"].map(
    {
        "google/gemma-2-2b": "Gemma 2 2B",
        "google/gemma-2-9b": "Gemma 2 9B",
        "google/gemma-2-27b": "Gemma 2 27B",
    }
)

# Add the number of model fitting iterations.
runs_configs_df["Model Fitting Iteration"] = runs_configs_df["dataset"].apply(
    src.analyze.determine_model_fitting_iteration_from_datasets_str
)

plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="line",
    x="Model Fitting Iteration",
    y="eval/loss",
    col="paradigm",
    col_order=["Replace", "Accumulate"],
    hue="Model",
    hue_order=["Gemma 2 2B", "Gemma 2 9B", "Gemma 2 27B"],
    palette="cool",
    legend="full",
    marker="o",
    markersize=15,
)
g.set(xlim=(0.5, 10.5), yscale="log")
g.set_axis_labels(y_var="Cross Entropy on Real Data (Test)", fontsize=20)
g.set_titles(col_template="{col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
for ax in g.axes.flat:
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs="auto", numticks=10))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:g}" if x != 1 else "")
    )
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=eval_loss_x=model_fitting_iteration_col=setting",
)
plt.show()

# extended_run_histories_df = runs_histories_df.merge(
#     runs_configs_df[["run_id", "Model Fitting Iteration"]],
#     left_on="run_id",
#     right_on="run_id",
# )
#
#
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


print("Finished running notebooks/01_sft_language_model.py")
