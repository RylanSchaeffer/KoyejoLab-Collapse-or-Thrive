import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.scale
import matplotlib.ticker
import matplotlib.transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")


# Enable LaTeX rendering.
# https://stackoverflow.com/a/23856968
# plt.rc('text', usetex=True)
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Computer Modern"
# Can add more commands to this list
plt.rcParams["text.latex.preamble"] = "\n".join([r"\usepackage{amsmath}"])
# Increase font size.
plt.rcParams["font.size"] = 23


def save_plot_with_multiple_extensions(
    plot_dir: str, plot_filename: str, use_tight_layout: bool = True
):
    if use_tight_layout:
        # Ensure that axis labels don't overlap.
        plt.gcf().tight_layout()

    extensions = [
        "pdf",
        "png",
    ]
    for extension in extensions:
        plot_path = os.path.join(plot_dir, plot_filename + f".{extension}")
        print(f"Plotted {plot_path}")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
