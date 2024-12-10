# Copyright 2020-2023 The TFP CausalImpact Authors
# Copyright 2014 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# without warranties or conditions of any kind, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for processing posterior results, including computing quantiles and
transforming posterior samples from model space back to original data space.
"""

from typing import List, Text, Tuple, Union

from causalimpact_gibbs import data as cid
import numpy as np
import pandas as pd
import tensorflow as tf


def calculate_trajectory_quantiles(
        trajectories: pd.DataFrame,
        column_prefix: Text = "predicted",
        quantiles: Tuple[float, float] = (0.025, 0.975)
) -> pd.DataFrame:
    """
    Calculate timepoint-wise quantiles of trajectory samples.

    This function computes specified quantiles for each time point across multiple
    trajectory samples. It can be applied to posterior predictive samples or cumulative
    trajectories, providing uncertainty bounds for the inferred trajectories.

    ### Parameters

    - **trajectories** (`pd.DataFrame`):
      DataFrame containing trajectory samples. Each row corresponds to a time point,
      and each column corresponds to a different sample. The index is typically a
      DatetimeIndex.

    - **column_prefix** (`str`, optional):
      Prefix for the resulting quantile columns in the output DataFrame. For example,
      use `"cumulative"` for cumulative trajectories to obtain columns named
      `"cumulative_lower"` and `"cumulative_upper"`. Defaults to `"predicted"`.

    - **quantiles** (`Tuple[float, float]`, optional):
      A tuple specifying the lower and upper quantiles to compute, both values
      between 0 and 1. Defaults to `(0.025, 0.975)` for 95% credible intervals.

    ### Returns

    `pd.DataFrame`:
    A DataFrame containing the calculated quantiles for each time point. The
    returned DataFrame has the same index as `trajectories` and includes two new
    columns named with the specified `column_prefix`, for example `"predicted_lower"`
    and `"predicted_upper"`.

    ### Raises

    - **ValueError**:
      If `quantiles` does not contain exactly two values or if they are not both
      within the (0, 1) interval.
    """
    # Validate the quantiles input
    if len(quantiles) != 2 or not all(0 < q < 1 for q in quantiles):
        raise ValueError("`quantiles` must be a tuple of two floats between 0 and 1.")

    quantile_suffixes = ["lower", "upper"]
    quantile_column_names = [f"{column_prefix}_{suffix}" for suffix in quantile_suffixes]

    # Calculate the quantiles for each time point
    quantiles_calculated = trajectories.quantile(q=quantiles, axis=1)

    # Transpose to match desired output format (time points as rows)
    quantiles_df = quantiles_calculated.transpose()

    # Assign the appropriate column names
    quantiles_df.columns = quantile_column_names

    # Ensure the index matches the original trajectories' index
    quantiles_df.index = trajectories.index

    return quantiles_df


import numpy as np
import pandas as pd
from typing import List, Text


def process_posterior_quantities(ci_data: cid.CausalImpactData,
                                 vals_to_process: np.ndarray,
                                 col_names: List[Text]) -> pd.DataFrame:
    """
    Process posterior samples by undoing scaling and reshaping into a time-indexed DataFrame.

    This function:
    - Undoes any scaling applied to the outcome or covariates before modeling.
    - Reshapes the input array so that rows correspond to time points and columns to
      samples (or derived quantities).
    - Returns a `pd.DataFrame` indexed by time, compatible with the original dataset.

    Parameters
    ----------
    ci_data : cid.CausalImpactData
        The `CausalImpactData` object containing information about the original data,
        including pre- and post-intervention periods and scaling.

    vals_to_process : np.ndarray
        An array of shape `[num_time_points, num_samples]`, where `num_samples` is the
        number of posterior samples or derived posterior quantities and `num_time_points`
        is the number of time steps in the full pre-post period.

    col_names : List[Text]
        Column names for the resulting DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame with rows corresponding to time points (covering both pre- and
        post-intervention periods) and columns named according to `col_names`. The
        DataFrame index is a time-based index derived from the original `CausalImpactData`.
    """
    # Undo scaling if it was applied
    if ci_data.standardize_data:
        vals_to_process = ci_data.outcome_scaler.inverse_transform(vals_to_process)

    # Ensure vals_to_process is a NumPy array
    if isinstance(vals_to_process, tf.Tensor):
        vals_to_process = vals_to_process.numpy()
    elif not isinstance(vals_to_process, np.ndarray):
        raise TypeError(f"vals_to_process must be a NumPy array or TensorFlow Tensor, got {type(vals_to_process)}")

    # Ensure vals_to_process is 2D
    if vals_to_process.ndim == 1:
        vals_to_process = vals_to_process.reshape(1, -1)

    # Determine the expected shape
    time_index = ci_data.normalized_pre_intervention_data.index.union(
        ci_data.normalized_after_pre_intervention_data.index).sort_values()
    num_time_points = len(time_index)
    num_samples = len(col_names)

    # Check if the data needs to be transposed
    if vals_to_process.shape[0] == num_samples and vals_to_process.shape[1] == num_time_points:
        # Transpose to [num_time_points, num_samples]
        vals_to_process = vals_to_process.T
    elif vals_to_process.shape[0] == num_time_points and vals_to_process.shape[1] == num_samples:
        # Already in correct orientation
        pass
    else:
        raise ValueError(
            f"Unexpected shape of vals_to_process: {vals_to_process.shape}, expected ({num_time_points}, {num_samples}) or ({num_samples}, {num_time_points})")

    # Final sanity check
    if vals_to_process.shape[0] != len(time_index) or vals_to_process.shape[1] != len(col_names):
        raise ValueError(
            f"Shape of vals_to_process {vals_to_process.shape} does not match expected (len(index), len(col_names)) = ({len(time_index)}, {len(col_names)}).")

    return pd.DataFrame(vals_to_process, columns=col_names, index=time_index)
