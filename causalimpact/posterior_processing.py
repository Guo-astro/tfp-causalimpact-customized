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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for working with (results from) the posterior."""

from typing import List, Text, Tuple, Union

from causalimpact import data as cid
import numpy as np
import pandas as pd

import pandas as pd
from typing import Text, Tuple, Union


def calculate_trajectory_quantiles(
        trajectories: pd.DataFrame,
        column_prefix: Text = "predicted",
        quantiles: Tuple[float, float] = (0.025, 0.975)
) -> pd.DataFrame:
  """
  Calculates timepoint-wise quantiles of trajectory samples.

  This function computes specified quantiles for each time point across multiple trajectory samples. It is applicable to both posterior and cumulative predictions, providing uncertainty bounds for the trajectories.

  Args:
      trajectories (pd.DataFrame):
          DataFrame containing trajectory samples. The DataFrame should have a
          DatetimeIndex where each row represents a time point and each column
          represents a different sample.

      column_prefix (str, optional):
          Prefix for the resulting quantile columns in the output DataFrame.
          For example, use "cumulative" for cumulative trajectories to obtain
          columns named "cumulative_lower" and "cumulative_upper". Defaults to "predicted".

      quantiles (Tuple[float, float], optional):
          A tuple specifying the lower and upper quantiles to compute. Each value
          should be between 0 and 1. Defaults to (0.025, 0.975), representing the
          2.5th and 97.5th percentiles.

  Returns:
      pd.DataFrame:
          A DataFrame containing the calculated quantiles for each time point. The
          DataFrame retains the original DatetimeIndex and includes two new columns
          named using the provided `column_prefix`, such as "predicted_lower" and
          "predicted_upper".

  Raises:
      ValueError:
          If `quantiles` does not contain exactly two values or if they are not
          within the (0, 1) interval.
  """

  # Validate the quantiles input
  if len(quantiles) != 2 or not all(0 < q < 1 for q in quantiles):
    raise ValueError("`quantiles` must be a tuple of two floats between 0 and 1.")

  # Define the suffixes for the quantile columns
  quantile_suffixes = ["lower", "upper"]
  quantile_column_names = [f"{column_prefix}_{suffix}" for suffix in quantile_suffixes]

  # Calculate the quantiles across samples for each time point
  quantiles_calculated = trajectories.quantile(q=quantiles, axis=1)

  # Transpose the result to have time points as rows
  quantiles_df = quantiles_calculated.transpose()

  # Assign the new column names based on the prefix and quantile suffixes
  quantiles_df.columns = quantile_column_names

  # Ensure the index matches the original trajectories' index
  quantiles_df.index = trajectories.index

  return quantiles_df


def process_posterior_quantities(ci_data: cid.CausalImpactData,
                                 vals_to_process: np.ndarray,
                                 col_names: List[Text]) -> pd.DataFrame:
  """Process posterior quantities by undoing any scaling and reshaping.

  This function assumes that the input np.ndarray `vals_to_process` has one or
  more rows corresponding to posterior samples and columns corresponding to
  time points. IMPORTANT: this function assumes that the time points correspond
  to the full time period, pre- and post-period combined. The function does the
  following:
  * undoes any scaling, if needed
  * reshapes so that rows correspond to time points and columns to samples.
  * reformats as a pd.DataFrame with a DatetimeIndex and appropriate column
    names.

  Args:
    ci_data: an instance of a cid.CausalImpactData object.
    vals_to_process: the input array.
    col_names: list of column names to use in the output.

  Returns:
    pd.DataFrame with rows corresponding to time points and columns
    named according to `col_names`.
  """
  # If the data used for modeling were scaled, first undo the scaling.
  if ci_data.standardize_data:
    vals_to_process = ci_data.outcome_scaler.inverse_transform(vals_to_process)
  # Transpose so that rows are again time points and columns are samples.
  vals_to_process = np.transpose(vals_to_process)

  # Format as pd.DataFrame and add the appropriate DatetimeIndex. Use the union
  # of the pre/post data in case the pre/post periods don't cover the full
  # data time period.
  index = ci_data.normalized_pre_intervention_data.index.union(
      ci_data.normalized_after_pre_intervention_data.index).sort_values()
  return pd.DataFrame(vals_to_process, columns=col_names, index=index)
