import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import seaborn as sns

import src.analyze
import src.data
import src.plot


data_dir, results_dir = src.analyze.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)


dataset_name_list = [
    "blobs",
    "circles",
    "moons",
    "swiss_roll",
]


# Create mock pandas dataframes.
dfs_dict = {}
for dataset_name in dataset_name_list:
    data, labels = src.data.create_dataset_for_kde(
        num_samples_per_iteration=2500,
        data_config_dict={"dataset_name": dataset_name, "dataset_kwargs": {}},
    )

    # Create a pandas dataframe.
    if dataset_name == "swiss_roll":
        df = pd.DataFrame(data[:, :3], columns=["x", "y", "z"])
    else:
        df = pd.DataFrame(data[:, :2], columns=["x", "y"])

    # Add the labels.
    df["Label"] = labels

    if dataset_name == "blobs":
        df["Dataset"] = "Blobs"
    elif dataset_name == "circles":
        df["Dataset"] = "Circles"
    elif dataset_name == "moons":
        df["Dataset"] = "Moons"
    elif dataset_name == "swiss_roll":
        df["Dataset"] = "Swiss Roll"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    dfs_dict[dataset_name] = df


# Plot the data.
plt.close()
fig = plt.figure(figsize=(5, 16))
for row_idx, (dataset_name, df) in enumerate(dfs_dict.items()):
    if dataset_name != "swiss_roll":
        ax = fig.add_subplot(4, 1, row_idx + 1)
        ax.scatter(
            df["x"],
            df["y"],
            c=df["Label"],
            cmap="viridis",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax = fig.add_subplot(4, 1, row_idx + 1, projection="3d")
        ax.scatter(
            df["x"],
            df["y"],
            df["z"],
            c=df["z"],
            cmap="viridis",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
    # # Add title on the right side of the axis
    # if isinstance(ax, Axes3D):
    #     ax.text2D(
    #         1.02,
    #         0.5,
    #         df["Dataset"].iloc[0],
    #         rotation=270,
    #         verticalalignment="center",
    #         transform=ax.transAxes,
    #     )
    #     # Remove z-axis label.
    #     # ax.set_zticks([])
    #
    # else:
    #     ax.text(
    #         1.02,
    #         0.5,
    #         df["Dataset"].iloc[0],
    #         rotation=270,
    #         verticalalignment="center",
    #         transform=ax.transAxes,
    #     )
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="datasets_with_3d_swiss_roll",
)
# plt.show()
print("Finished running 04_visualize_kde_datasets.py")
