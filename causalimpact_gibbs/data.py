# data.py
"""Class for storing and preparing data for modeling."""
from typing import Optional, Text, Tuple, Union

from causalimpact_gibbs import indices
from causalimpact_gibbs import standardize
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp


class CausalImpactData:
    """Class for storing and preparing data for modeling.

    This class handles all of the data-related functions of CausalImpact. It
    makes sure the input data is given in an appropriate format (as a pandas
    DataFrame or convertible to pd.DataFrame), checks that the given
    outcome/feature column names exist, and splits the data into the pre-period
    and post-period based on the given treatment start time OR tuples defining
    the start/end points of the pre/post periods.

    If the pre- and post-periods do not cover the entire timespan of
    the data, the excluded portions will be used for plotting but NOT for
    fitting the model or calculating impact statistics.

    Note that the default is to standardize the pre-period data to have mean zero
    and standard deviation one. The standardization is done for each column
    separately, and then applied to the post-period data using each column's pre-
    period mean and standard deviation. The mean and standard deviation for the
    outcome column are stored so that the inferences can be back-transformed to
    the original scale.


    Attributes:
      data: Pandas DataFrame of timeseries data.
      pre_intervention_period: Start and end value in data.index for pre-intervention.
      post_intervention_period: Start and end value in data.index for post-intervention.
      target_col: Timeseries being modeled. Defaults to first column of
        `data`.
      feature_columns: Subset of data.columns used as covariates. `None` in case
        there are no covariates. Defaults to all non-outcome columns (or `None` if
        there are none).
      standardize_data: Boolean: Whether covariates and outcome were scaled to
        have 0 mean and 1 standard deviation.
      pre_intervention_data: Subset of `data` from `pre_period`. This is unscaled.
      after_pre_intervention_data: Subset of `data` from after the `pre_period`. The time
        between pre-period and post-period should still be forecasted to make
        accurate post-period predictions. Additionally, users are interested in
        after post-period predictions. This is unscaled.
      num_steps_forecast: Number of elements (including NaN) to forecast for,
        including the post-period and time between pre-period and post-period.
      model_pre_data: Scaled subset of `data` from `pre_period` used for fitting
        the model.
      model_after_pre_data: Scaled subset of `data` from `post_period` used for
        fitting the model.
      outcome_scaler: A `standardize.Scaler` object used to transform outcome
        data.
      feature_ts: A pd.DataFrame of the scaled data over just the feature columns.
      pre_intervention_target_ts: A tfp.sts.MaskedTimeSeries instance of the outcome data from the
        `pre_period`.
    """

    def __init__(self,
                 data: Union[pd.DataFrame, pd.Series],
                 pre_intervention_period: Tuple[indices.InputDateType, indices.InputDateType],
                 post_intervention_period: Tuple[indices.InputDateType, indices.InputDateType],
                 drop_post_period_nan=True,
                 target_col_name: Optional[Text] = None,
                 standardize_data=True,
                 dtype=tf.float32):
        """Constructs a `CausalImpactData` instance.

        Args:
          data: Pandas `DataFrame` containing an outcome time series and optional
            feature time series.
          pre_intervention_period: Pre-period start and end (see InputDateType).
          post_intervention_period: Post-period start and end (see InputDateType).
          target_col_name: String giving the name of the outcome column in `data`. If
            not specified, the first column in `data` is used.
          standardize_data: If covariates and output should be standardized.
          dtype: The dtype to use throughout computation.
        """
        # This is a no-op in case data is a pd.DataFrame. It is common enough to
        # pass a pd.Series that this is useful here.
        data = pd.DataFrame(data)
        self.pre_intervention_period, self.post_intervention_period = indices.parse_and_validate_date_data(
            data=data, pre_intervention_period=pre_intervention_period,
            post_intervention_period=post_intervention_period)
        self.data, self.target_col, self.feature_columns = (
            _validate_data_and_columns(data, target_col_name))
        del data  # To insure the unfiltered DataFrame is not used again.
        self.standardize_data = standardize_data
        self.pre_intervention_data = self.data.loc[(self.data.index >= self.pre_intervention_period[0])
                                                   & (self.data.index <= self.pre_intervention_period[1])]
        # after_pre_data intentionally includes everything after the end of the
        # pre-period since the time between pre- and post-period needs to be
        # accounted for and we actually want to see predictions after the post
        # period.
        self.after_pre_intervention_data = self.data.loc[self.data.index > self.pre_intervention_period[1]]
        self.num_steps_forecast = len(self.after_pre_intervention_data.index)

        if self.standardize_data:
            scaler = standardize.Scaler().fit(self.pre_intervention_data)
            self.outcome_scaler = standardize.Scaler().fit(
                self.pre_intervention_data[self.target_col])
            self.normalized_pre_intervention_data = scaler.transform(self.pre_intervention_data)
            self.normalized_after_pre_intervention_data = scaler.transform(self.after_pre_intervention_data)
        else:
            self.outcome_scaler = None
            self.normalized_pre_intervention_data = self.pre_intervention_data
            self.normalized_after_pre_intervention_data = self.after_pre_intervention_data
        if drop_post_period_nan:
            self.normalized_after_pre_intervention_data = self.normalized_after_pre_intervention_data.dropna()

        normalized_pre_intervention_target_tf_tensor = tf.convert_to_tensor(
            self.normalized_pre_intervention_data[self.target_col], dtype=dtype)
        self.pre_intervention_target_ts = tfp.sts.MaskedTimeSeries(
            time_series=normalized_pre_intervention_target_tf_tensor,
            is_missing=tf.math.is_nan(normalized_pre_intervention_target_tf_tensor))
        if self.feature_columns is not None:
            # Here we have to use the FULL time series so that the post-period
            # feature data can be used for forecasting.
            normalized_pre_intervention_features = self.normalized_pre_intervention_data[self.feature_columns]
            normalized_post_intervention_features = self.normalized_after_pre_intervention_data[self.feature_columns]
            self.normalized_whole_period_features = pd.concat(
                [normalized_pre_intervention_features, normalized_post_intervention_features], axis=0)
            self.normalized_whole_period_features["intercept_"] = 1.
        else:
            self.normalized_whole_period_features = None


def _validate_data_and_columns(data: pd.DataFrame,
                               target_column_name: Optional[str]):
    """Validates data and sets defaults for feature and outcome columns.

    By default, the first column of the dataframe will be used as the target_column,
    and the rest will be used as features, but these can instead be provided.

    Args:
      data: Input dataframe for analysis.
      target_column_name: Optional string to use for the target_column_name.

    Raises:
      KeyError: if `outcome_column` is not in the data.
      ValueError: if `outcome_column` is constant.

    Returns:
      The validated (possibly default) data, outcome column, and feature columns.
    """

    # Check outcome column -- if not specified, default is the first column.
    if target_column_name is None:
        target_column_name = data.columns[0]
    if target_column_name not in data.columns:
        raise KeyError(f"Specified `outcome_column` ({target_column_name}) not found "
                       f"in data")

    # Make sure outcome column is not constant
    if data[target_column_name].std(skipna=True, ddof=0) == 0:
        raise ValueError("Input response cannot be constant.")

    # Feature columns are all those other than the output column. Use
    # `original_column_order` to keep track of the
    # original column order, since set(data.columns) reorders the
    # columns, which leads to problems later when subsetting and transforming.
    if data.shape[1] <= 1:
        feature_columns = None
    else:
        original_column_order = data.columns
        column_differences = set(data.columns).difference([target_column_name])
        feature_columns = [
            col for col in original_column_order if col in column_differences
        ]
    data = data[[target_column_name] + (feature_columns or [])]
    if data[target_column_name].count() < 3:  # Series.count() is for non-NaN values.
        raise ValueError("Input data must have at least 3 observations.")
    if data[feature_columns or []].isna().values.any():
        raise ValueError("Input data cannot have any missing values.")
    if not data.dtypes.map(pd.api.types.is_numeric_dtype).all():
        raise ValueError("Input data must contain only numeric values.")

    return data, target_column_name, feature_columns
