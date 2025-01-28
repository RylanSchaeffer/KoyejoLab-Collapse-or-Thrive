import ast
import matplotlib.colors
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


data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


definitions_df = pd.read_csv(
    "data/icml_2025_position/definitions.csv",
)
# Create a nicely formatted "Author (Date)" column.
definitions_df["Author_Date"] = (
    definitions_df["Paper Authors"] + " (" + definitions_df["Date Released"] + ")"
)

# Drop the "Paper Name" column.
definitions_df = definitions_df.drop(columns=["Paper Name"])


# List of definition columns to include in the heatmap
definition_columns = [
    "Catastrophic Increase of Population Risk",
    "Any Increase of Population Risk",
    "Asymptotically Diverging Risk",
    "Change in Scaling Law",
    "Collapsing Variance",
    "Entanglement of Real Data Mode(s) over Time",
    "Disappearance of Real Tail Data over Time",
    "Appearance of Hallucinated Data Over Time",
]

plt.close()

# Create the figure with two subplots
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(20, 15),
    gridspec_kw={
        "height_ratios": [1, len(definition_columns)],
        "hspace": 0.1,
        # "width_ratios": [0.95, 0.05],
    },
    constrained_layout=True,
)

# Create the upper heatmap. This heatmap will show whether each paper has an explicit definition.
ax0 = axes[0]
upper_data = pd.DataFrame(
    [definitions_df["Explicit Definition"].values],
    columns=definitions_df["Author_Date"],
)
upper_data.index = ["Explicit Definition"]
upper_numeric_df = (upper_data == "Yes").astype(int)
upper_annot_df = upper_data.applymap(lambda x: "Yes" if x == "Yes" else "")
upper_heatmap = sns.heatmap(
    upper_numeric_df,
    # cmap=["#f0f0f0", "#4169e1"],  # Grey, Blue
    cmap=["#f0f0f0", "#33a532"],  # Grey, Green
    cbar=False,
    # cbar_kws={"label": ""},
    # cbar_ax=axes[0, 1],
    annot=upper_annot_df,
    fmt="",
    xticklabels=False,  # Hide x labels for upper plot
    ax=ax0,
)
# # Manually set the colorbar ticks and labels for upper heatmap
# colorbar_upper = upper_heatmap.collections[0].colorbar
# colorbar_upper.set_ticks([0.25, 0.75])
# colorbar_upper.set_ticklabels(["No", "Yes"])
# Rotate y-axis labels for better readability
ax0.tick_params(axis="y", rotation=0)
ax0.set_xlabel("")
ax0.set_title("Is Model Collapse Explicitly Defined?")

# Lower Heatmap (Definition Components) ==============================
ax1 = axes[1]
heatmap_data = definitions_df[definition_columns].T
heatmap_data.columns = definitions_df["Author_Date"]
# Create numeric mapping for coloring
lower_numeric_df = pd.DataFrame(
    0, index=heatmap_data.index, columns=heatmap_data.columns
)
lower_annot_df = pd.DataFrame(
    "", index=heatmap_data.index, columns=heatmap_data.columns
)
for i in heatmap_data.index:
    for j in heatmap_data.columns:
        value = str(heatmap_data.loc[i, j]).lower()
        if pd.isna(value) or value == "nan":
            lower_numeric_df.loc[i, j] = 0
            lower_annot_df.loc[i, j] = ""
        elif "explicitly" in value:
            lower_numeric_df.loc[i, j] = 2
            lower_annot_df.loc[i, j] = "E"
        elif "implicitly" in value:
            lower_numeric_df.loc[i, j] = 1
            lower_annot_df.loc[i, j] = "I"
        else:
            lower_numeric_df.loc[i, j] = 0
            lower_annot_df.loc[i, j] = "•" if value.lower() in ["true", "yes"] else ""
# Create the lower heatmap
lower_heatmap = sns.heatmap(
    lower_numeric_df,
    cmap=["#f0f0f0", "#ffd700", "#4169e1"],
    cbar=False,
    # cbar_kws={"label": ""},
    # cbar_ax=axes[1, 1],  # pass your own Axes
    annot=lower_annot_df,
    fmt="",
    xticklabels=True,
    yticklabels=True,
    ax=ax1,
)
# # Manually set the colorbar ticks and labels for upper heatmap
# colorbar_lower = lower_heatmap.collections[0].colorbar
# colorbar_lower.set_ticks([0.33, 1.0, 1.67])
# colorbar_lower.set_ticklabels(["Unused", "Implicit", "Explicit"])
# Rotate y-axis yticklabels for better readability
ax1.tick_params(axis="y", rotation=0)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
ax1.set_xlabel("")
ax1.set_title(r"Which Definitions of Model Collapse Are Used?")

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="definitions_meta_analysis",
    use_tight_layout=False,
    # use_tight_layout=True,
)
plt.show()

print("Finished notebooks/90_definitions_schematic")
