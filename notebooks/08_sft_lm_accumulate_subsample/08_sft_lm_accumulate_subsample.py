import matplotlib.colors
import matplotlib.pyplot as plt
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
    "y4oue58c",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration1 (Part 1)
    "3cn97hta",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration1 (Part 2)
    "4ec1abqn",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration2
    "z0u305ey",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration3
    "3a7l9hl6",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration4
    "zmpfqtvj",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration5
    "j9y6fl6a",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration6
    "8ewj0gxn",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration7
    "y7gsymxb",  # HelpSteer2   Gemma2-2B   Paradigm=Accumulate-Subsample   Iteration8
    "n2ren5e9",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration1
    "5i6bj5re",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration2
    "n8t9j5ee",  # HelpSteer2   Gemma2-2B   Paradigm=Replace                Iteration3
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

runs_configs_df = runs_configs_df.rename(
    columns={
        "num_samples_per_iteration": "Num. Samples per Iteration",
    }
)

# Add the number of model fitting iterations.
runs_configs_df["Model Fitting Iteration"] = runs_configs_df["dataset"].apply(
    src.analyze.determine_model_fitting_iteration_from_datasets_str
)

runs_configs_df["Num. Samples per Iteration"] = 12500

runs_configs_df["Task"] = "SFT Language Models"

plt.close()
g = sns.relplot(
    data=runs_configs_df,
    kind="line",
    x="Model Fitting Iteration",
    y="eval/loss",
    col="paradigm",
    col_order=["Replace", "Accumulate-Subsample", "Accumulate"],
    row="Task",
    palette="cool",
    legend="full",
    hue="Num. Samples per Iteration",
    # hue_norm=matplotlib.colors.LogNorm(),
    # marker="o",
    # markersize=15,
    facet_kws={"sharex": True, "sharey": True, "margin_titles": True},
)
g.set(yscale="log")
g.set_axis_labels(y_var="Cross Entropy on Real Data (Test)", fontsize=20)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=eval_loss_x=model_fitting_iteration_col=setting",
)
# plt.show()
