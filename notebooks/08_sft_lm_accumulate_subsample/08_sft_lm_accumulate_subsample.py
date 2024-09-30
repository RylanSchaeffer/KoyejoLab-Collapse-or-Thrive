import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

import src.analyze
import src.plot


refresh = False
# refresh = True

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=refresh,
)

wandb_username = "rylan"
wandb_sweep_ids = [
    "y4oue58c",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration1 (Part 1)
    "3cn97hta",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration1 (Part 2)
    "4ec1abqn",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration2
    "z0u305ey",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration3
    "3a7l9hl6",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration4
    "zmpfqtvj",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration5
    "j9y6fl6a",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration6
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

# Add the number of model fitting iterations.
runs_configs_df["Model Fitting Iteration"] = runs_configs_df["dataset"].apply(
    src.analyze.determine_model_fitting_iteration_from_datasets_str
)

runs_configs_df["Task"] = "SFT of LMs"

plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="line",
    x="Model Fitting Iteration",
    y="eval/loss",
    col="Setting",
    col_order=["Replace", "Accumulate-Subsample", "Accumulate"],
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
