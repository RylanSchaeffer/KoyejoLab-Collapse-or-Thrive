import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

import src.analyze
import src.plot


##############################################################################
# HELPER FUNCTION FOR HEX-CELL “HEATMAP” (NO COLORBAR)
##############################################################################
def draw_hex_heatmap(
    ax,
    data,
    color_list,
    annotations=None,
    flip_rows=True,
    xlabels=None,
    ylabels=None,
    xrotation=90,
):
    """
    Draw a hex 'heatmap' on Axes `ax` from a 2D integer `data` array:
      - data[i,j] indexes into color_list
      - annotations[i,j] is an optional string to plot at the cell center
      - flip_rows=True => row 0 at the top, typical 'heatmap' style
      - xlabels/ylabels => if not None, show ticklabels
      - xrotation => rotation (degrees) for x-tick labels
    """
    cmap = ListedColormap(color_list)
    boundaries = np.arange(-0.5, len(color_list) + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)

    rows, cols = data.shape
    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            color = cmap(norm(val))
            # Flip row indices so row 0 is at the top
            y_coord = (rows - 1 - i) if flip_rows else i
            x_coord = j
            # Draw a hex
            hex_patch = mpatches.RegularPolygon(
                (x_coord + 0.5, y_coord + 0.5),
                numVertices=6,
                radius=0.45,
                orientation=np.radians(30),
                edgecolor="white",
                facecolor=color,
            )
            ax.add_patch(hex_patch)
            # Annotation
            if annotations is not None:
                ann = annotations[i, j]
                if isinstance(ann, str) and ann.strip():
                    ax.text(
                        x_coord + 0.5,
                        y_coord + 0.5,
                        ann,
                        ha="center",
                        va="center",
                        fontsize=11,
                        color="black",
                    )

    # Adjust axes so hexes fit neatly
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")

    # Apply xlabels and ylabels if provided
    if xlabels is not None:
        ax.set_xticks(np.arange(cols) + 0.5)
        ax.set_xticklabels(xlabels, rotation=xrotation, ha="right")
    else:
        ax.set_xticks([])

    if ylabels is not None:
        ax.set_yticks(np.arange(rows) + 0.5)
        if flip_rows:
            ax.set_yticklabels(ylabels[::-1])
        else:
            ax.set_yticklabels(ylabels)
    else:
        ax.set_yticks([])


##############################################################################
# MAIN SCRIPT
##############################################################################

data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

definitions_df = pd.read_csv("data/icml_2025_position/definitions.csv")
definitions_df["Author_Date"] = (
    definitions_df["Paper Authors"] + " (" + definitions_df["Date Released"] + ")"
)
definitions_df = definitions_df.drop(columns=["Paper Name"])

definition_columns = [
    "Catastrophic Increase of Population Risk",
    "Any Increase of Population Risk",
    "Asymptotically Diverging Population Risk",
    "Change in Scaling Law",
    "Collapsing Variance",
    "Entanglement of Real Data Mode(s)",
    "Disappearance of Real Tail Data",
    "Appearance of Hallucinated Data",
]

plt.close()

# We increase the bottom subplot's ratio so it is much bigger than the top
fig, (ax0, ax1) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(20, 10),
    gridspec_kw={
        "height_ratios": [
            1,
            len(definition_columns),
        ],  # bigger ratio for the bottom plot
        "hspace": 0.05,  # reduce vertical space between subplots
    },
    constrained_layout=True,
)

##############################################################################
# UPPER HEX-HEATMAP (Explicit Definition)
##############################################################################
upper_data = pd.DataFrame(
    [definitions_df["Explicit Definition?"].values],
    columns=definitions_df["Author_Date"],
)
upper_data.index = ["Explicit Definition?"]


# 0 => grey, 1 => blue
def convert_str_to_int(s) -> int:
    if s == "Yes":
        return 2
    elif s == "No":
        return 1
    else:
        return 0


def convert_int_to_str(i) -> str:
    if i == 2:
        return "Y"
    elif i == 1:
        return "N"
    else:
        return "N/A"


upper_numeric_df = upper_data.applymap(convert_str_to_int)
upper_annot_df = upper_numeric_df.applymap(convert_int_to_str)

upper_numeric_np = upper_numeric_df.values
upper_annot_np = upper_annot_df.values

draw_hex_heatmap(
    ax=ax0,
    data=upper_numeric_np,
    # color_list=["#f0f0f0", "#4169e1"],  # Grey, Blue
    # color_list=["#f0f0f0", "#33a532"],  # Grey, Green
    color_list=["#f0f0f0", "#e67c82", "#9aeb8a"],  # Grey, Red, Green
    annotations=upper_annot_np,
    xlabels=None,  # NO x tick labels at the top
    ylabels=upper_numeric_df.index,
    xrotation=90,
)
ax0.set_title("Is Model Collapse Explicitly Defined?")

##############################################################################
# LOWER HEX-HEATMAP (Definition Components)
##############################################################################
heatmap_data = definitions_df[definition_columns].T
heatmap_data.columns = definitions_df["Author_Date"]

lower_numeric_df = pd.DataFrame(
    0, index=heatmap_data.index, columns=heatmap_data.columns
)
lower_annot_df = pd.DataFrame(
    "", index=heatmap_data.index, columns=heatmap_data.columns
)

for i in heatmap_data.index:
    for j in heatmap_data.columns:
        val_str = str(heatmap_data.loc[i, j]).lower()
        if pd.isna(val_str) or val_str == "nan":
            lower_numeric_df.loc[i, j] = 0
            lower_annot_df.loc[i, j] = ""
        elif "explicitly" in val_str:
            lower_numeric_df.loc[i, j] = 2
            lower_annot_df.loc[i, j] = "E"
        elif "implicitly" in val_str:
            lower_numeric_df.loc[i, j] = 1
            lower_annot_df.loc[i, j] = "I"
        else:
            if val_str in ["true", "yes"]:
                lower_annot_df.loc[i, j] = "•"
            lower_numeric_df.loc[i, j] = 0

# 0 => grey, 1 => gold, 2 => blue
lower_numeric_np = lower_numeric_df.values
lower_annot_np = lower_annot_df.values

draw_hex_heatmap(
    ax=ax1,
    data=lower_numeric_np,
    color_list=["#f0f0f0", "#ffd700", "#c0e9ff"],  # Grey, Yellow, Blue
    annotations=lower_annot_np,
    xlabels=lower_numeric_df.columns,  # We DO want xlabels here
    ylabels=lower_numeric_df.index,
    xrotation=45,
    flip_rows=True,
)
ax1.set_title("Which Definitions of Model Collapse Are Used?")
ax1.set_ylabel("Definitions of Model Collapse")
# # Add text
# fig.text(
#     0.075,
#     0.81,
#     r"\underline{Definitions of Model Collapse}",
#     # ha="center",
#     # va="center",
#     color="black",
# )

# # Optionally reduce extra whitespace around the figure
# plt.subplots_adjust(left=0.2, right=0.95, top=0.92, bottom=0.1)

src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="definitions_meta_analysis_hex_no_cbar",
    use_tight_layout=False,
)
plt.show()

print("Finished notebooks/90_definitions_schematic0")
