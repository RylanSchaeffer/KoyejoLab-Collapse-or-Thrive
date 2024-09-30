import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

import src.analyze
import src.plot


# refresh = False
refresh = False

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=refresh,
)

wandb_username = "jkazdan"
wandb_sweep_ids = [
    "q3vd9gyn",  # HelpSteer2   Gemma2-2B   Data=Original   Iteration1
    "2cvqmk2v",  # HelpSteer2   Gemma2-2B   Data=Replace    Iteration2
    "wtr77bli",
    "6s09ojgi",
    "8ha71vqm",
    "nqd2zmqg",
    "63o3uyjm",
    "utw2dy7b",
    "xqjudpc0",
    "2z9f726i",
    "3ryjlwpj",
    "no35bjlm",
    "hjshv3r0",
    "nxzbezmg",
    "oqr34ktf",
    "zc52fldc",
    "tph8nlpx",
    "ccac0yx5",
    "r32e7rwu",
]
# for the accumulate data
# wandb_sweep_ids = [
#     "q3vd9gyn",  # HelpSteer2   Gemma2-2B   Data=Original   Iteration1
#     "3ryjlwpj",  # HelpSteer2   Gemma2-2B   Data=Replace    Iteration2
#     "no35bjlm",
#     "hjshv3r0",
#     "nxzbezmg",
#     "oqr34ktf",
#     "nqd2zmqg",
#     "tph8nlpx"
# ]
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

# Add the number of model fitting iterations.
runs_configs_df["Model Fitting Iteration"] = runs_configs_df["dataset"].apply(
    src.analyze.determine_model_fitting_iteration_from_datasets_str
)
runs_configs_df["Setting"] = runs_configs_df["dataset"].apply(
    src.analyze.determine_setting_from_datasets_str
)

runs_histories_df: pd.DataFrame = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="ft_collapse",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb_username,
    wandb_run_history_samples=100000000,  # Make sure we grab _all_ the data.
)

runs_configs_df, runs_histories_df = src.analyze.duplicate_real_data_runs(
    runs_configs_df=runs_configs_df,
    runs_histories_df=runs_histories_df,
)


plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="line",
    x="Model Fitting Iteration",
    y="eval/loss",
    col="Setting",
    col_order=["Replace", "Accumulate"],
    marker="o",
    markersize=15,
)
g.set_axis_labels(y_var="Eval Cross Entropy on Real Data", fontsize=20)
g.set_titles(col_template="{col_name}")
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=eval_loss_x=model_fitting_iteration_col=setting",
)
# plt.show()

extended_run_histories_df = runs_histories_df.merge(
    runs_configs_df[["run_id", "Model Fitting Iteration"]],
    left_on="run_id",
    right_on="run_id",
)


plt.close()
g = sns.relplot(
    data=extended_run_histories_df,
    kind="line",
    x="train/epoch",
    y="eval/loss",
    col="Setting",
    hue="Model Fitting Iteration",
)
g.set_yticklabels(fontsize=10)
g.set_axis_labels("Epoch", "Eval Cross Entropy on Real Data")
g.set_titles("{col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=eval_loss_x=epoch_col=setting_hue=model_fitting_iteration",
)
# plt.show()

# Visualize each individual run's learning curve.
runs_learning_curves_dir = os.path.join(results_dir, "learning_curves_per_run")
os.makedirs(runs_learning_curves_dir, exist_ok=True)
for run_id, run_history_df in extended_run_histories_df.groupby("run_id"):
    # extended_run_histories_df.loc[
    #     run_history_df.index, "eval/loss_smoothed"
    # ] = run_history_df["eval/loss"].rolling(window=10).mean()
    plt.close()
    sns.lineplot(
        data=run_history_df,
        x="train/epoch",
        y="eval/loss",
    )
    plt.title(f"Run ID: {run_id}")
    src.plot.save_plot_with_multiple_extensions(
        plot_dir=runs_learning_curves_dir,
        plot_filename=f"y=eval_loss_x=epoch_run_id={run_id}",
    )
    # plt.show()


print("Finished running notebooks/01_sft_language_model.py")
