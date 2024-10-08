import ast
import hashlib
import os

import numpy as np
import pandas as pd
import pyarrow
import requests
import time
from typing import Dict, List, Optional, Set, Tuple, Union
import wandb
from tqdm import tqdm

import src.globals


def determine_model_fitting_iteration_from_datasets_str(datasets_str: str) -> int:
    if datasets_str == "nvidia/HelpSteer2":
        return 1
    num_datasets = len(datasets_str.split(","))
    if num_datasets > 1:
        # In this case, we are accumulating.
        # Thus, the number of datasets is the number of model-fitting iterations.
        return num_datasets
    elif "iter" in datasets_str:
        return int(datasets_str.split("iter")[1].split("_", maxsplit=1)[0]) + 1
    else:
        raise ValueError("How the hell did you end up here?")


def determine_data_ratio_from_datasets_str(datasets_str: str) -> int:
    if datasets_str == "nvidia/HelpSteer2":
        return 1
    num_datasets = len(datasets_str.split(","))
    if num_datasets > 1:
        # In this case, we are accumulating.
        # Thus, the number of datasets is the number of model-fitting iterations.
        return num_datasets
    elif "iter" in datasets_str:
        return int(datasets_str.split("iter")[1].split("_", maxsplit=1)[0]) + 1
    else:
        raise ValueError("How the hell did you end up here?")


def determine_setting_from_datasets_str(datasets_str: str) -> str:
    num_datasets = len(datasets_str.split(","))
    if num_datasets == 1:
        if datasets_str == "nvidia/HelpSteer2":
            setting = "Original"
        else:
            setting = "Replace"
    elif num_datasets > 1:
        setting = "Accumulate"
    else:
        raise ValueError("How the hell did you end up here?")
    return setting


def download_wandb_project_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: str = None,
    filetype: str = "csv",
) -> pd.DataFrame:
    assert filetype in {"csv", "feather", "parquet"}

    # Hash because otherwise too long.
    filename = "sweeps=" + ",".join(sweep_ids)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    runs_configs_df_path = os.path.join(
        data_dir, hashed_filename + f"_runs_configs.{filetype}"
    )
    if refresh or not os.path.isfile(runs_configs_df_path):
        print(f"Creating {runs_configs_df_path} anew.")

        # Download sweep results
        api = wandb.Api(timeout=600)

        if wandb_username is None:
            wandb_username = api.viewer.username

        sweep_results_list = []
        for sweep_id in sweep_ids:
            try:
                sweep = api.sweep(f"{wandb_username}/{wandb_project_path}/{sweep_id}")
                for run in tqdm(sweep.runs):
                    # .summary contains the output keys/values for metrics like accuracy.
                    #  We call ._json_dict to omit large files
                    summary = run.summary._json_dict

                    # .config contains the hyperparameters.
                    #  We remove special values that start with _.
                    summary.update(
                        {k: v for k, v in run.config.items() if not k.startswith("_")}
                    )

                    summary.update(
                        {
                            "State": run.state,
                            "Sweep": run.sweep.id if run.sweep is not None else None,
                            "run_id": run.id,
                        }
                    )
                    # .name is the human-readable name of the run.
                    summary.update({"run_name": run.name})
                    sweep_results_list.append(summary)
            except:
                pass

        runs_configs_df = pd.DataFrame(sweep_results_list)
        runs_configs_df.reset_index(inplace=True, drop=True)

        # Save to disk.
        runs_configs_df.to_csv(
            runs_configs_df_path.replace(filetype, "csv"), index=False
        )
        try:
            runs_configs_df.to_feather(
                runs_configs_df_path.replace(filetype, "feather")
            )
        except BaseException:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass
        try:
            runs_configs_without_model_generations_kwargs_df = runs_configs_df.drop(
                columns=["model_generation_kwargs"]
            )
            runs_configs_without_model_generations_kwargs_df.to_parquet(
                runs_configs_df_path.replace(filetype, "parquet"), index=False
            )
        except BaseException:
            # pyarrow.lib.ArrowNotImplementedError: Cannot write struct type 'model_generation_kwargs' with no child field to Parquet. Consider adding a dummy child field.
            pass

        print(f"Regenerated and wrote {runs_configs_df_path} to disk.")
        del runs_configs_df

    print(f"Reading {runs_configs_df_path} from disk.")
    if filetype == "csv":
        runs_configs_df = pd.read_csv(runs_configs_df_path)
    elif filetype == "feather":
        runs_configs_df = pd.read_feather(runs_configs_df_path)
    elif filetype == "parquet":
        runs_configs_df = pd.read_parquet(runs_configs_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
    print(f"Loaded {runs_configs_df_path} from disk.")

    # Keep only finished runs
    finished_runs = runs_configs_df["State"] == "finished"
    print(
        f"% of successfully finished runs: {100.0 * finished_runs.mean()} ({finished_runs.sum()} / {len(finished_runs)})"
    )

    if finished_only:
        runs_configs_df = runs_configs_df[finished_runs]

        # Check that we don't have an empty data frame.
        assert len(runs_configs_df) > 0

        # Ensure we aren't working with a slice.
        runs_configs_df = runs_configs_df.copy()

    return runs_configs_df


def download_wandb_project_runs_histories(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    wandb_run_history_samples: int = 10000,
    refresh: bool = False,
    wandb_username: Optional[str] = None,
    filetype: str = "csv",
    nrows_to_read: Optional[int] = None,
) -> pd.DataFrame:
    assert filetype in {"csv", "feather", "parquet"}

    # Hash because otherwise too long.
    filename = "sweeps=" + ",".join(sweep_ids)
    hashed_filename = hashlib.md5(filename.encode()).hexdigest()
    runs_histories_df_path = os.path.join(
        data_dir, hashed_filename + f"_runs_histories.{filetype}"
    )
    if refresh or not os.path.isfile(runs_histories_df_path):
        # Download sweep results
        api = wandb.Api(timeout=6000)

        if wandb_username is None:
            wandb_username = api.viewer.username

        runs_histories_list = []
        print("Downloading runs' histories...")
        for iteration, sweep_id in enumerate(sweep_ids):
            sweep = api.sweep(f"{wandb_username}/{wandb_project_path}/{sweep_id}")
            for run in tqdm(sweep.runs):
                # https://community.wandb.ai/t/run-history-returns-different-values-on-almost-each-call/2431/4

                history = None
                for num_attempts in range(5):
                    try:
                        history = run.history(
                            samples=wandb_run_history_samples,
                        )
                        break
                    except (requests.exceptions.HTTPError, wandb.errors.CommError):
                        print("Retrying...")
                        time.sleep(3)

                # Skip this run if the API doesn't let us download it.
                if history is None or history.empty:
                    continue

                # Drop generation columns because these are meaty.
                generation_columns = [
                    col for col in history.columns if "generation" in col
                ]
                history.drop(columns=generation_columns, inplace=True)
                history["run_id"] = run.id
                history["model_fitting_iteration"] = iteration
                runs_histories_list.append(history)

        assert len(runs_histories_list) > 0
        runs_histories_df = pd.concat(runs_histories_list, sort=False)

        runs_histories_df.sort_values(["run_id"], ascending=True, inplace=True)
        runs_histories_df.reset_index(inplace=True, drop=True)

        # Save all three because otherwise this is a pain in the ass.
        runs_histories_df.to_csv(
            runs_histories_df_path.replace(filetype, "csv"), index=False
        )
        try:
            runs_histories_df.to_feather(
                runs_histories_df_path.replace(filetype, "feather")
            )
        except BaseException:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass
        try:
            runs_histories_df.to_parquet(
                runs_histories_df_path.replace(filetype, "parquet"), index=False
            )
        except pyarrow.lib.ArrowInvalid:
            # pyarrow.lib.ArrowInvalid: ("Could not convert 'NaN' with type str: tried to convert to double", 'Conversion failed for column loss/score_model=claude3opus with type object')
            pass
        print(f"Wrote {runs_histories_df_path} to disk")
        del runs_histories_df

    print(f"Loading {runs_histories_df_path} from disk.")
    if filetype == "csv":
        runs_histories_df = pd.read_csv(runs_histories_df_path, nrows=nrows_to_read)
    elif filetype == "feather":
        runs_histories_df = pd.read_feather(runs_histories_df_path)
    elif filetype == "parquet":
        runs_histories_df = pd.read_parquet(runs_histories_df_path)
    else:
        raise ValueError(f"Invalid filetype: {filetype}")
    print(f"Loaded {runs_histories_df_path} from disk.")

    return runs_histories_df


def duplicate_real_data_runs(
    runs_configs_df: pd.DataFrame,
    runs_histories_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # First, add "Setting" from runs_configs_df to runs_histories_df .
    # Otherwise, this will cause problems down the line.
    runs_histories_df["Setting"] = runs_histories_df["run_id"].map(
        runs_configs_df.set_index("run_id")["Setting"]
    )

    # Find runs with "Model Fitting Iteration" == 1.
    real_data_runs_ids = runs_configs_df[
        runs_configs_df["Model Fitting Iteration"] == 1
    ]["run_id"].unique()

    # Duplicate those rows in runs_configs_df.
    duplicate_real_data_runs_configs_df = runs_configs_df[
        runs_configs_df["run_id"].isin(real_data_runs_ids)
    ].copy()

    # Set the copies' "Setting" to "Replace".
    duplicate_real_data_runs_configs_df["Setting"] = "Replace"

    # Set the original rows' "Setting" to "Accumulate".
    runs_configs_df.loc[
        runs_configs_df["run_id"].isin(real_data_runs_ids), "Setting"
    ] = "Accumulate"

    # Combine back in.
    runs_configs_df = pd.concat(
        [runs_configs_df, duplicate_real_data_runs_configs_df],
        ignore_index=True,
    ).reset_index(drop=True)

    # Duplicate those rows in runs_histories_df.
    duplicate_real_data_runs_histories_df = runs_histories_df[
        runs_histories_df["run_id"].isin(real_data_runs_ids)
    ].copy()

    # Set the copies' "Setting" to "Replace".
    duplicate_real_data_runs_histories_df["Setting"] = "Replace"

    # Set the original rows' "Setting" to "Accumulate".
    runs_histories_df.loc[
        runs_histories_df["run_id"].isin(real_data_runs_ids), "Setting"
    ] = "Accumulate"

    # Combine back in.
    runs_histories_df = pd.concat(
        [runs_histories_df, duplicate_real_data_runs_histories_df],
        ignore_index=True,
    ).reset_index(drop=True)

    # Check that no more Original remains.
    assert not (runs_configs_df["Setting"] == "Original").any()
    assert not (runs_histories_df["Setting"] == "Original").any()

    return runs_configs_df, runs_histories_df


def extract_function(
    col_name: str, key_in_dict: Optional[str] = None, new_col_name: Optional[str] = None
):
    if isinstance(x, dict):
        return x[key_in_dict]
    elif x.isnan():
        return np.NaN
    else:
        return ast.literal_eval(x)[key_in_dict]


# parse data config blob into cols
def extract_key_value_from_df_col(
    df: pd.DataFrame,
    col_name: str,
    key_in_dict: Optional[str] = None,
    new_col_name: Optional[str] = None,
):
    if new_col_name is None:
        new_col_name = key_in_dict
    df[new_col_name] = df[col_name].apply(
        lambda x: (
            x[key_in_dict]
            if isinstance(x, dict)
            else (np.NaN if isinstance(x, float) else ast.literal_eval(x)[key_in_dict])
        )
    )
    return df


def setup_notebook_dir(
    notebook_dir: str,
    refresh: bool = False,
) -> Tuple[str, str]:
    # Declare paths.
    data_dir = os.path.join(notebook_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    results_dir = os.path.join(notebook_dir, "results")
    if refresh:
        import shutil

        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir
