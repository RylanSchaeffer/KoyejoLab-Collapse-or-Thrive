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


rate_of_approaching_zero_df = pd.DataFrame(
    {
        "Model-Fitting Iteration": [1, 2, 3, 4, 5] * 2,
        "Setting": ["Replace"] * 5 + ["Accumulate"] * 5,
        "Percentage of Real Data": [1.0, 0.0, 0.0, 0.0, 0]
        + [1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0],
    }
)
rate_of_approaching_zero_df["Percentage of Real Data"] *= 100.0

plt.close()
g = sns.relplot(
    data=rate_of_approaching_zero_df,
    kind="line",
    x="Model-Fitting Iteration",
    y="Percentage of Real Data",
    col="Setting",
    # markers=True,
    marker="o",  # Specifies marker style
    markersize=10,  # Controls marker size
    color="gray",
    height=6,  # Height of each facet in inches
    aspect=1.2,  # Width/height ratio for each facet
)
g.set_titles(col_template="{col_name}")
g.set(xlim=(0.75, 5.25))
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_filename="y=percent_real_data_x=model_fitting_iteration_col=setting",
)
plt.show()

print("Finished notebooks/91_rate_of_approaching_zero")
