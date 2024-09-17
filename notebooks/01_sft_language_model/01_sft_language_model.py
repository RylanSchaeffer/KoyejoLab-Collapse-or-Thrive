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

wandb_username = "jkazdan"
# wandb_sweep_ids = [
#     "q3vd9gyn",  # HelpSteer2   Gemma2-2B   Data=Original   Iteration1
#     "2cvqmk2v",  # HelpSteer2   Gemma2-2B   Data=Replace    Iteration2
#     "wtr77bli",
#     "6s09ojgi",
#     "8ha71vqm",
#     "nqd2zmqg",
#     "63o3uyjm",
#     "utw2dy7b",
# ]
# for the accumulate data
wandb_sweep_ids = [
    "q3vd9gyn",  # HelpSteer2   Gemma2-2B   Data=Original   Iteration1
    "3ryjlwpj",  # HelpSteer2   Gemma2-2B   Data=Replace    Iteration2
    "no35bjlm",
    "hjshv3r0",
]
runs_configs_df: pd.DataFrame = src.analyze.download_wandb_project_runs_configs(
    wandb_project_path="ft_collapse",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb_username,
)

# After this, we now have a column called "dataset" in runs_configs_df.
runs_configs_df = src.analyze.extract_key_value_from_df_col(
    df=runs_configs_df,
    col_name="data_config",
    key_in_dict="dataset",
    new_col_name="dataset",
)


runs_histories_df: pd.DataFrame = src.analyze.download_wandb_project_runs_histories(
    wandb_project_path="ft_collapse",
    data_dir=data_dir,
    sweep_ids=wandb_sweep_ids,
    refresh=refresh,
    wandb_username=wandb_username,
)

extended_run_histories_df = runs_histories_df.merge(
    runs_configs_df[["run_id", "dataset"]], left_on="run_id", right_on="run_id"
)

# for i in range(len(wandb_sweep_ids)):
#     extended_run_histories_df[extended_run_histories_df['wandb_sweep_ids'].eq(wandb_sweep_ids[i])]['model_fitting_iteration'] = i+1

# This is a basic demonstration.
plt.close()
g = sns.relplot(
    data=extended_run_histories_df,
    kind="line",
    x="train/epoch",
    y="eval/loss",
    # col="dataset",
    hue="model_fitting_iteration",
)
g.set_axis_labels("Epoch", "Eval Loss on Real Data")
g.set_titles("{col_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="sft_language_model_loss_vs_epoch_by_dataset",
)
plt.show()
print("Finished running 01_sft_language_model.py")


# print(extended_run_histories_df.keys())
# plt.close()
# g = sns.relplot(
#     data=extended_run_histories_df,
#     x="model_fitting_iteration",
#     y="eval/loss",
#     hue="_step",
# )
# g.set_axis_labels("Fitting Iteration", "Eval Loss on Real Data")
# g.set_titles("{col_name}")
# sns.move_legend(g, "upper left", bbox_to_anchor=(1.0, 1.0))
# src.plot.save_plot_with_multiple_extensions(
#     plot_dir=results_dir,
#     plot_filename="sft_language_model_eval_vs_fitting_iteration",
# )
# plt.show()
