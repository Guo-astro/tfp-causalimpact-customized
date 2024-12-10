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

"""Plotting causalimpact_gibbs results."""
from pprint import pprint
from typing import Any, Union, Dict

import altair as alt
import numpy as np
import japanize_matplotlib
import arviz as az

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import logging

from causalimpact_gibbs import CausalImpactAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Define a mapping from internal variable names to descriptive titles
VARIABLE_TITLES = {
    'level': 'Trend Level',
    'level_scale': 'Trend Level Scale',
    'observation_noise_scale': 'Observation Noise Scale',
    'weights': 'Weights',
    'seasonal_drift_scales': 'Seasonal Drift Scales',
    'seasonal_levels': 'Seasonal Levels'
}


def _generate_diagnostic_plots(
        inference_data: az.InferenceData,
        convergence_diagnostics: Dict[str, Any],
        fig_width_in: float,
        fig_height_in: float,
        generate_diagnostic_plots: bool
):
    """
    Generate diagnostic plots for the posterior samples using ArviZ.

    This function generates trace plots and autocorrelation plots for all relevant
    variables in the InferenceData object, adding detailed legends and indicating
    convergence status based on R-hat statistics. Additionally, it annotates the plots
    with median, mean, max, and min values of R-hat for each variable.

    Parameters
    ----------
    inference_data : az.InferenceData
        The ArviZ InferenceData object containing posterior samples and other diagnostics.

    convergence_diagnostics : dict
        A dictionary containing convergence diagnostics such as R-hat and ESS.

    Returns
    -------
    None
        Displays the diagnostic plots.
    """
    # Define a mapping for variable titles (optional)
    VARIABLE_TITLES = {
        'level': 'Local Level',
        'level_scale': 'Level Scale',
        'observation_noise_scale': 'Observation Noise Scale',
        'weights': 'Regression Weights',
        'seasonal_drift_scales': 'Seasonal Drift Scales',
        'seasonal_levels': 'Seasonal Levels'
    }

    # List of variables to generate diagnostics for
    variables_to_plot = ['level', 'level_scale', 'observation_noise_scale', 'weights']

    # Add seasonal variables if present
    seasonal_variables = ['seasonal_drift_scales', 'seasonal_levels']
    variables_to_plot.extend(seasonal_variables)

    # Define convergence threshold
    RHAT_THRESHOLD = 1.1

    # Iterate over each variable and generate plots if data is present
    for var in variables_to_plot:
        # Check if the variable exists in the posterior group
        if var in inference_data.posterior:
            var_data = inference_data.posterior[var]

            # Check if the data variable has any non-zero dimensions
            if var_data.dims and all(dim_size > 0 for dim_size in var_data.sizes.values()):
                try:
                    # Retrieve the descriptive title; default to the variable name if not found
                    var_title = VARIABLE_TITLES.get(var, var)

                    # Retrieve R-hat for the variable
                    rhat: np.ndarray | None = convergence_diagnostics.get('rhat', {}).get(var, None)

                    if rhat is not None:
                        # Compute R-hat statistics
                        rhat_mean = np.mean(rhat)
                        rhat_median = np.median(rhat)
                        rhat_min = np.min(rhat)
                        rhat_max = np.max(rhat)

                        # Compute posterior statistics
                        posterior_mean = var_data.mean().values
                        posterior_median = var_data.median().values

                        # Determine if R-hat indicates convergence
                        is_converged = np.all(rhat < RHAT_THRESHOLD)
                        convergence_status = "Converged" if is_converged else "Not Converged"
                        color = 'green' if is_converged else 'red'

                        # Log the R-hat statistics
                        logger.info(
                            f"Variable '{var_title}': R-hat Mean={rhat_mean:.3f}, "
                            f"Median={rhat_median:.3f}, Min={rhat_min:.3f}, "
                            f"Max={rhat_max:.3f} - {convergence_status}"
                        )

                        # Create a formatted string for annotations
                        rhat_stats = (
                            f"R-hat Statistics:\n"
                            f"Mean = {rhat_mean:.3f}\n"
                            f"Median = {rhat_median:.3f}\n"
                            f"Min = {rhat_min:.3f}\n"
                            f"Max = {rhat_max:.3f}\n\n"
                            f"Posterior Statistics:\n"
                            f"Mean = {posterior_mean:.3f}\n"
                            f"Median = {posterior_median:.3f}"
                        )

                        if generate_diagnostic_plots:
                            # --- Trace Plot ---
                            az.plot_trace(var_data, var_names=[var], compact=True)
                            plt.suptitle(f"Trace Plot for {var_title}", fontsize=16)

                            # Add detailed legends
                            handles, labels = plt.gca().get_legend_handles_labels()
                            if handles and labels:
                                plt.legend(
                                    handles,
                                    [f"Chain {label}" for label in labels],
                                    title='Chains',
                                    loc='upper right'
                                )

                            # Annotate the plot with convergence statistics
                            plt.figtext(
                                0.95, 0.95, rhat_stats,
                                ha="right", va="top", fontsize=10,
                                bbox=dict(
                                    facecolor='white',
                                    alpha=0.8,
                                    edgecolor=color,
                                    boxstyle='round,pad=0.5'
                                )
                            )

                            # --- Autocorrelation Plot ---
                            az.plot_autocorr(var_data, var_names=[var], figsize=(fig_width_in, fig_height_in))
                            # --- Posterior Plot ---
                            az.plot_posterior(var_data, var_names=[var], figsize=(fig_width_in, fig_height_in))
                            # --- Forest Plot ---
                            az.plot_forest(var_data, var_names=[var], ess=True, r_hat=True,
                                           figsize=(fig_width_in, fig_height_in))
                            plt.suptitle(f"Autocorrelation Plot for {var_title}")

                            # Add detailed legends
                            handles, labels = plt.gca().get_legend_handles_labels()
                            if handles and labels:
                                plt.legend(
                                    handles,
                                    [f"Chain {label}" for label in labels],
                                    title='Chains',
                                    loc='upper right'
                                )

                            # Annotate the plot with convergence statistics
                            plt.figtext(
                                0.95, 0.95, rhat_stats,
                                ha="right", va="top", fontsize=10,
                                bbox=dict(
                                    facecolor='white',
                                    alpha=0.8,
                                    edgecolor=color,
                                    boxstyle='round,pad=0.5'
                                )
                            )


                    else:
                        logger.warning(f"R-hat values for variable '{var_title}' are missing.")

                except Exception as e:
                    logger.warning(f"Failed to plot diagnostics for variable '{var_title}': {e}")
            else:
                logger.info(f"Skipping plots for variable '{var}' as it has no data.")
        else:
            logger.info(f"Variable '{var}' not found in posterior samples. Skipping plots.")

    logger.info("Diagnostic plots generation completed.")


def _draw_matplotlib_plot(data_frame: pd.DataFrame, ci: CausalImpactAnalysis = None, generate_diagnostic_plots=True,
                          **plot_options):
    """
    Generate a customized Matplotlib figure with four subplots:
    1. Observed vs. Mean
    2. Pointwise Effect
    3. Cumulative Effect
    4. Trace Plot of a Selected Parameter (if ci is provided)

    Parameters
    ----------
    data_frame : pd.DataFrame
        The DataFrame containing the data to be plotted.
    ci : CausalImpactAnalysis, optional
        The CausalImpactAnalysis object with posterior samples to plot the trace.
    plot_options : dict, optional
        Dictionary of plotting options. Same as before plus additional
        options for trace plot as needed.
    """

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # ----------------------------- Helper Functions -----------------------------
    def process_y_formatter_units(y_formatter_unit_options, subplot_labels):
        if isinstance(y_formatter_unit_options, str):
            return {label: y_formatter_unit_options for label in subplot_labels}
        elif isinstance(y_formatter_unit_options, list):
            if len(y_formatter_unit_options) != len(subplot_labels):
                raise ValueError("Length of y_formatter_unit list must match number of y_labels.")
            return dict(zip(subplot_labels, y_formatter_unit_options))
        elif isinstance(y_formatter_unit_options, dict):
            return y_formatter_unit_options
        else:
            raise TypeError("y_formatter_unit must be a string, list, or dict.")

    def create_y_axis_formatter(format_option, unit):
        if format_option == 'millions':
            return lambda x, pos: f'{x * 1e-6:.1f}{unit}'
        elif format_option == 'thousands':
            return lambda x, pos: f'{x * 1e-3:.1f}{unit}'
        elif callable(format_option):
            return lambda x, pos: f'{format_option(x, pos)}{unit}'
        else:
            return lambda x, pos: f'{x}{unit}'

    def add_period_markers(ax, df, labels):
        pre_start = df["pre_period_start"].iloc[0]
        pre_end = df["pre_period_end"].iloc[0]
        post_start = df["post_period_start"].iloc[0]
        post_end = df["post_period_end"].iloc[0]

        if (df["time"] < pre_start).any():
            ax.axvline(pre_start, color="grey", linestyle="--", label=labels['pre_period_start'])

        if ((df["time"] > pre_end) & (df["time"] < post_start)).any():
            ax.axvline(pre_end, color="grey", linestyle="--", label=labels['pre_period_end'])

        ax.axvline(post_start, color="grey", linestyle="--", label=labels['post_period_start'])

        if (df["time"] > post_end).any():
            ax.axvline(post_end, color="grey", linestyle="--", label=labels['post_period_end'])

        handles, legend_labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(legend_labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(),
                  loc="upper left", fontsize="small", frameon=False)

    # ----------------------------- Extract Plot Options -----------------------------
    chart_width_px = plot_options.get("chart_width", 800)
    chart_height_px = plot_options.get("chart_height", 600)
    dpi = 100
    # Now we have 4 subplots (instead of 3), adjust height accordingly
    fig_width_in = chart_width_px / dpi
    fig_height_in = (chart_height_px * 4) / dpi

    x_label = plot_options.get("x_label", "Time")
    y_labels = plot_options.get("y_labels", ["Observed", "Pointwise Effect", "Cumulative Effect"])
    plot_title = plot_options.get("title", "")
    title_font_size = plot_options.get("title_font_size", 14)
    axis_label_font_size = plot_options.get("axis_title_font_size", 12)

    y_formatter_option = plot_options.get("y_formatter", "millions")
    y_formatter_unit_option = plot_options.get("y_formatter_unit", "dollar")

    legend_labels = plot_options.get("legend_labels", {})
    legend = {
        'mean': legend_labels.get("mean", "Mean"),
        'observed': legend_labels.get("observed", "Observed"),
        'pointwise': legend_labels.get("pointwise", "Pointwise"),
        'cumulative': legend_labels.get("cumulative", "Cumulative"),
        'pre_period_start': legend_labels.get("pre_period_start", "Pre-Period Start"),
        'pre_period_end': legend_labels.get("pre_period_end", "Pre-Period End"),
        'post_period_start': legend_labels.get("post_period_start", "Post-Period Start"),
        'post_period_end': legend_labels.get("post_period_end", "Post-Period End"),
    }

    # Add one more label for the trace plot if you want distinct formatting
    extended_y_labels = y_labels + ["Trace Plot"]
    y_formatter_units = process_y_formatter_units(y_formatter_unit_option, extended_y_labels)
    default_unit = "dollar"

    # ----------------------------- Create Subplots -----------------------------
    fig, axes = plt.subplots(
        nrows=3,  # now we have 4 rows
        ncols=1,
        figsize=(fig_width_in, fig_height_in),
        sharex=False,  # Trace plot might not share the same x-axis
        constrained_layout=True
    )

    if plot_title:
        fig.suptitle(plot_title, fontsize=title_font_size)

    # Assign each subplot a y-label
    subplot_labels = extended_y_labels
    for ax, y_label in zip(axes, subplot_labels):
        ax.grid(True, linestyle='--', alpha=0.7)
        unit = y_formatter_units.get(y_label, default_unit)
        formatter = create_y_axis_formatter(y_formatter_option, unit)
        ax.yaxis.set_major_formatter(FuncFormatter(formatter))
        ax.set_ylabel(y_label, fontsize=axis_label_font_size,
                      fontweight="bold", labelpad=10)

    # Add period markers only to the first 3 subplots which deal with time series data
    for ax in axes[:3]:
        add_period_markers(ax, data_frame, legend)

    # Set the common x-axis label on the last time-series subplot (3rd one)
    axes[2].set_xlabel(x_label, fontsize=axis_label_font_size, fontweight="bold")
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # ----------------------------- Plot Data on Subplots -----------------------------

    # Subplot 1: Observed vs Mean
    observed_mask = (data_frame["scale"] == "original") & (data_frame["stat"] == "observed")
    mean_mask = (data_frame["scale"] == "original") & (data_frame["stat"] == "mean")

    observed_data = data_frame[observed_mask]
    mean_data = data_frame[mean_mask]

    axes[0].plot(mean_data["time"], mean_data["value"],
                 label=legend['mean'], color='blue')
    axes[0].plot(observed_data["time"], observed_data["value"],
                 label=legend['observed'], color='orange')
    axes[0].fill_between(
        observed_data["time"],
        observed_data["lower"],
        observed_data["upper"],
        color='gray',
        alpha=0.2
    )
    axes[0].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small", frameon=False)

    # Subplot 2: Pointwise Effect
    pointwise_mask = (data_frame["scale"] == "point_effects") & (data_frame["stat"] == "mean")
    pointwise_data = data_frame[pointwise_mask]

    axes[1].plot(pointwise_data["time"], pointwise_data["value"],
                 label=legend['pointwise'], color='green')
    axes[1].fill_between(
        pointwise_data["time"],
        pointwise_data["lower"],
        pointwise_data["upper"],
        color='lightgreen',
        alpha=0.2
    )
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=1)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small", frameon=False)

    # Subplot 3: Cumulative Effect
    cumulative_mask = (data_frame["scale"] == "cumulative_effects") & (data_frame["stat"] == "mean")
    cumulative_data = data_frame[cumulative_mask]

    axes[2].plot(cumulative_data["time"], cumulative_data["value"],
                 label=legend['cumulative'], color='red')
    axes[2].fill_between(
        cumulative_data["time"],
        cumulative_data["lower"],
        cumulative_data["upper"],
        color='salmon',
        alpha=0.2
    )
    axes[2].axhline(0, color="grey", linestyle="--", linewidth=1)
    axes[2].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small", frameon=False)

    # ---------------------- Subplot 4: Trace Plot of MCMC Samples --------------------
    # Check if ci and posterior samples are available
    if ci is not None:
        diagnostics = {}
        inference_data = az.from_dict(posterior=ci.convergence_diagnostics)

        # Compute R-hat and ESS
        rhat = az.rhat(inference_data)
        ess = az.ess(inference_data)
        diagnostics["rhat"] = rhat
        diagnostics["ess"] = ess
        coords = inference_data.posterior.coords
        num_computational_draws = coords["draw"].shape[0] * coords["chain"].shape[0]
        print(f"Number of computational draws: {num_computational_draws}")
        _generate_diagnostic_plots(inference_data, diagnostics, fig_width_in, fig_height_in, generate_diagnostic_plots)

        summary = az.summary(inference_data)
        pprint(summary)
    # Align y-labels across subplots for a cleaner look
    fig.align_ylabels(axes)

    return fig


def plot(ci_model: CausalImpactAnalysis, generate_diagnostic_plots: bool, **kwargs) -> Union[alt.Chart, Any]:
    """Main plotting function.

    Args:
      ci_model: CausalImpactAnalysis object, after having called
        `fit_causalimpact`.
      **kwargs: arguments for modifying plot defaults:
        static_plot - whether to return the standard CausalImpact plot as a
          static plot (default) or an interactive plot.
        backend - literal["altair","matplotlib"] to use for generating the figure
        alpha - float for determining confidence level for uncertainty intervals
          when quantile_based_intervals=False.
        show_median - whether to draw posterior median predictions in addition to
          the posterior mean. Only applies if "median" was a specified aggregation
          given in evaluate().
        use_std_intervals - whether to draw uncertainty intervals based on
          quantiles (default) or use a normal approximation based on the standard
          deviation.
        chart_width - integer for chart width in pixels.
        chart_height - integer for chart height in pixels.
        axis_title_font_size - integer for axis title font size. Default = 18.
        axis_label_font_size - integer for axis title font size. Default = 16.
        strip_title_font_size - integer for facet label font size. Default = 18.

    Returns:
      alt.Chart plot object
    """

    # Process kwargs.
    plot_params = {
        "static_plot": True,
        "backend": "matplotlib",
        "alpha": 0.05,
        "show_median": False,
        "use_std_intervals": False,
        "chart_width": 600,
        "chart_height": 200,
        "axis_label_font_size": 16,
        "strip_title_font_size": 20,
        "x_label": "Date",
        "y_labels": ["Observed", "Pointwise Effect", "Cumulative Effect"],
        "title": "Interrupted Time Series Analysis Over Time",
        "title_font_size": 16,
        "axis_title_font_size": 12,
        "y_formatter": "millions",
        "axes2_legend_label": "Cumulative",
        "axes1_legend_label": "Pointwise",
        "axes0_legend_label_mean": "Mean",
        "axes0_legend_label_observed": "Observed",
        "y_formatter_unit": "dollar",
        "legend_labels": {
            "mean": "Mean",
            "observed": "Observed",
            "pointwise": "Pointwise",
            "cumulative": "Cumulative",
            "pre-period-start": "Pre-Period Start",
            "pre-period-end": "Pre-Period End",
            "post-period-start": "Post-Period Start",
            "post-period-end": "Post-Period End",
        },

    }
    if kwargs:
        for k, v in plot_params.items():
            plot_params[k] = kwargs.get(k, v)

    # Create the dataframe that will be used to create the plot.
    main_plot_df = _create_plot_df(ci_model.series, plot_params["alpha"])

    # If use_std_intervals=True, use std to draw the uncertainty intervals,
    # otherwise use the quantile-based intervals. We drop the unnecessary
    # observations instead of keeping the requested ones because dates outside of
    # the pre/post period will have NaN values for band_method.
    if plot_params["use_std_intervals"]:
        plot_df = main_plot_df.loc[main_plot_df["band_method"] != "quantile"]
    else:
        plot_df = main_plot_df.loc[main_plot_df["band_method"] != "std"]

    # Include median if requested.
    if plot_params["show_median"]:
        plot_df = plot_df.loc[plot_df["stat"] != "median"]
        plot_df["stat_pretty"] = pd.Categorical(
            plot_df["stat_pretty"], categories=["Observed", "Mean"], ordered=True
        )

    # Create the requested plot type.
    if plot_params["backend"] == "altair":
        if plot_params["static_plot"]:
            plot_df = plot_df.loc[(plot_df["stat"] != "median")]
            plt = _draw_classic_plot(plot_df, **plot_params)
        else:
            plt = _draw_interactive_plot(plot_df, **plot_params)
    elif plot_params["backend"] == "matplotlib":
        plt = _draw_matplotlib_plot(plot_df, ci=ci_model, generate_diagnostic_plots=generate_diagnostic_plots,
                                    **plot_params)
    else:
        raise ValueError(
            "backend must be one of 'altair' or 'matplotlib'. Got"
            f" {plot_params['backend']}."
        )
    return plt


def _create_plot_df(series: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Creates a dataframe for plotting impact inferences.

    This function generates data for visualizing impact inferences by plotting observed and predicted values with uncertainty bands, and faceting by scale.


    Args:
        series (pd.DataFrame): Output from sts_mod.evaluate(), containing observed and predicted outcomes, as well as absolute and cumulative impact estimates.
        alpha (float): Confidence level for standard deviation-based uncertainty intervals.

    Returns:
        DataFrame indexed by time for plotting.
        Columns:
            value (float): Values to plot as lines.
            upper (float): Upper uncertainty bounds.
            lower (float): Lower uncertainty bounds.
            scale (str): Scale of `value` ('original', 'pointwise', 'cumulative').
            stat (str): Statistic of `value` ('observed', 'posterior_mean', 'posterior_median').
            scale_prett (str): Formatted scale label for plots.
            stat_pretty (str): Formatted statistic label for plots.
            zero (float): Reference line for absolute and cumulative effect plots.
    """
    series["time"] = series.index

    # Create dataframes for each component of the plot (lines and uncertainty
    # bands, and standard devation-based uncertainty bands if requested).
    lines_df = _create_plot_component_df(series, "lines")
    bands_df = _create_plot_component_df(series, "bands")
    if any(["std" in col for col in series.columns]):
        std_df = _create_plot_component_df(series, "std", alpha)
        bands_df = pd.concat([bands_df, std_df], axis=0, sort=True)

    # Merge the component dataframes. Use left join because lines_df will contain
    # observed data outside of the pre/post period intervals, whereas bands_df
    # will not.
    plot_df = lines_df.merge(
        bands_df,
        on=[
            "time", "scale", "pre_period_start", "pre_period_end",
            "post_period_start", "post_period_end"
        ],
        how="left")

    # Add a zero column so we can plot a zero line for the absolute and
    # cumulative scales, but set it to np.nan for the original scale so it
    # doesn't get plotted.
    plot_df["zero"] = 0
    plot_df.loc[plot_df["scale"] == "original", "zero"] = np.nan

    # Make nicer versions of the scale and stat variables for use as plot labels.
    plot_df["scale_pretty"] = [
        "Original" if m == "original" else
        "Pointwise" if m == "point_effects" else "Cumulative"
        for m in plot_df["scale"]
    ]
    plot_df["scale_pretty"] = pd.Categorical(
        plot_df["scale_pretty"],
        categories=["Original", "Pointwise", "Cumulative"],
        ordered=True)
    plot_df["stat_pretty"] = plot_df["stat"].str.capitalize()
    plot_df["stat_pretty"] = pd.Categorical(
        plot_df["stat_pretty"],
        categories=["Observed", "Mean", "Median"],
        ordered=True)

    return plot_df


import pandas as pd
import tensorflow_probability as tfp


def _create_plot_component_df(series: pd.DataFrame,
                              component: str,
                              alpha: float = 0.05) -> pd.DataFrame:
    """
    Generates a dataframe for specific plot components based on impact estimates.

    This function transforms the impact estimate time series into a long-form dataframe
    suitable for plotting lines or uncertainty bands on different scales: original,
    absolute (pointwise), or cumulative. Depending on the `component` parameter, it
    structures the dataframe to include necessary statistics or uncertainty bounds.

    Columns:
        time (datetime): Time index.
        value (float): Y-axis values to be plotted.
        scale (str): Scale of the value ('original', 'absolute', 'cumulative').
        stat (str, optional): Type of statistic ('observed', 'mean', 'median') for "lines" component.
        upper (float, optional): Upper bound of uncertainty interval for "bands" or "std" components.
        lower (float, optional): Lower bound of uncertainty interval for "bands" or "std" components.
        band_method (str, optional): Method used to calculate uncertainty bands ('quantiles', 'std').

    Args:
        series (pd.DataFrame): Dataframe containing impact estimates, typically output from sts_mod.evaluate().
        component (str): Type of plot component to create ("lines", "bands", "std").
        alpha (float, optional): Significance level for confidence intervals. Defaults to 0.05.

    Returns:
        pd.DataFrame: Transformed dataframe indexed by time with columns tailored to the specified component.

    Raises:
        ValueError: If `component` is not one of 'lines', 'bands', or 'std'.
    """

    valid_components = {"lines", "bands", "std"}
    if component not in valid_components:
        raise ValueError(f"`component` must be one of {valid_components}. Got '{component}'.")

    # Define the base columns required for all components
    base_columns = [
        "time", "pre_period_start", "pre_period_end",
        "post_period_start", "post_period_end"
    ]

    # Extend columns based on the component type
    if component == "lines":
        required_columns = base_columns + ["mean", "median", "observed"]
    elif component == "bands":
        required_columns = base_columns + ["lower", "upper"]
    else:  # component == "std"
        required_columns = base_columns + ["mean", "std"]

    # Extract relevant columns from the series
    extracted_columns = [col for col in series.columns if any(stub in col for stub in required_columns)]
    filtered_df = series[extracted_columns]

    # Melt the dataframe to long format for easier manipulation
    melted_df = filtered_df.melt(
        id_vars=base_columns,
        var_name="scale_stat",
        value_name="value"
    )

    # Extract 'scale' by removing statistical suffixes
    statistical_suffixes = "_upper|_lower|_mean|_median|_std"
    melted_df["scale"] = melted_df["scale_stat"].str.replace(statistical_suffixes, "", regex=True)

    # Assign 'original' scale to observed or posterior statistics
    original_scale_keywords = "observed|posterior"
    melted_df.loc[melted_df["scale"].str.contains(original_scale_keywords, regex=True), "scale"] = "original"

    # Extract 'stat' by removing scale prefixes
    scale_prefixes = "posterior_|point_effects_|cumulative_effects_"
    melted_df["stat"] = melted_df["scale_stat"].str.replace(scale_prefixes, "", regex=True)

    # Drop the intermediate 'scale_stat' column
    melted_df.drop(columns=["scale_stat"], inplace=True)

    # Reshape for 'bands' and 'std' components to have separate 'upper' and 'lower' columns
    if component in {"bands", "std"}:
        pivot_columns = [
            "time", "scale", "pre_period_start", "pre_period_end",
            "post_period_start", "post_period_end"
        ]
        pivoted_df = melted_df.pivot_table(
            index=pivot_columns,
            columns="stat",
            values="value"
        ).reset_index()

        # Assign the band calculation method based on the component
        pivoted_df["band_method"] = "quantiles" if component == "bands" else "std"

        # For 'std' component, calculate the confidence intervals using the standard deviation
        if component == "std":
            z_score = tfp.distributions.Normal(0, 1).quantile(1 - alpha / 2).numpy()
            pivoted_df["lower"] = pivoted_df["mean"] - z_score * pivoted_df["std"]
            pivoted_df["upper"] = pivoted_df["mean"] + z_score * pivoted_df["std"]
            pivoted_df.drop(columns=["mean", "std"], inplace=True)

        filtered_df = pivoted_df
    else:
        filtered_df = melted_df

    return filtered_df


def _create_base_layers(plot_df: pd.DataFrame, **kwargs) -> dict:
    """
    Create base plot layers for impact inference visualizations.

    This helper function generates the foundational layers for both static and interactive plots, including:
    - Lines representing the data values.
    - Uncertainty bands around the lines.
    - Horizontal reference line at zero.
    - Vertical reference lines indicating the start and end of pre- and post-treatment periods.

    Args:
        plot_df (pd.DataFrame): Dataframe containing the data to plot, including time indices and period boundaries.
        **kwargs: Optional plot parameters.
            - chart_width (int): Width of the chart in pixels. Default is 600.
            - chart_height (int): Height of the chart in pixels. Default is 200.
            - axis_title_font_size (int): Font size for axis titles. Default is 18.
            - axis_label_font_size (int): Font size for axis labels. Default is 16.
            - strip_title_font_size (int): Font size for facet labels. Default is 20.

    Returns:
        dict: A dictionary containing the base plot layers:
            - "lines" (alt.Chart): Line chart of the data values.
            - "band" (alt.Chart): Area chart representing uncertainty bands.
            - "hline" (alt.Chart): Horizontal rule at zero.
            - "vlines" (dict): Dictionary of vertical rules marking period boundaries, keyed by period identifiers.

    Raises:
        KeyError: If required period boundary columns are missing in `plot_df`.
    """

    # Define default plotting parameters
    chart_width = kwargs.get("chart_width", 600)
    chart_height = kwargs.get("chart_height", 200)
    axis_title_font_size = kwargs.get("axis_title_font_size", 18)
    axis_label_font_size = kwargs.get("axis_label_font_size", 16)
    strip_title_font_size = kwargs.get("strip_title_font_size", 20)

    # Validate required columns in plot_df
    required_columns = {
        "pre_period_start", "pre_period_end",
        "post_period_start", "post_period_end"
    }
    missing_columns = required_columns - set(plot_df.columns)
    if missing_columns:
        raise KeyError(f"The following required columns are missing in plot_df: {missing_columns}")

    # Extract period boundaries from the first row
    periods = {
        "pre_period_start": plot_df.at[0, "pre_period_start"],
        "pre_period_end": plot_df.at[0, "pre_period_end"],
        "post_period_start": plot_df.at[0, "post_period_start"],
        "post_period_end": plot_df.at[0, "post_period_end"],
    }

    # Create the base line layer for the data values
    base_lines = alt.Chart(plot_df).mark_line().encode(
        x=alt.X("time:T", title="Time"),
        y=alt.Y("value:Q", scale=alt.Scale(zero=False), title="")
    ).properties(
        width=chart_width,
        height=chart_height
    )

    # Create the uncertainty band layer
    uncertainty_band = alt.Chart(plot_df).mark_area(opacity=0.3).encode(
        x=alt.X("time:T", title="Time"),
        y="upper:Q",
        y2="lower:Q"
    ).properties(
        width=chart_width,
        height=chart_height
    )

    # Create a horizontal reference line at zero
    horizontal_zero = alt.Chart(plot_df).mark_rule(color="red").encode(
        y=alt.Y("zero:Q")
    )

    # Initialize dictionary to hold vertical reference lines
    vertical_lines = {}

    # Define helper function to add a vertical line if condition is met
    def add_vline(key, condition, x_value):
        if condition:
            vertical_lines[key] = alt.Chart(plot_df).mark_rule(
                strokeDash=[5, 5],
                color="grey"
            ).encode(
                x=alt.X(x_value + ":T")
            )

    # Add vertical lines based on period boundaries
    add_vline(
        "pre_period_start",
        (plot_df["time"] < periods["pre_period_start"]).any(),
        "pre_period_start"
    )
    add_vline(
        "pre_period_end",
        ((plot_df["time"] > periods["pre_period_end"]) & (plot_df["time"] < periods["post_period_start"])).any(),
        "pre_period_end"
    )
    add_vline(
        "post_period_start",
        True,  # Always draw the start of the post-period
        "post_period_start"
    )
    add_vline(
        "post_period_end",
        (plot_df["time"] > periods["post_period_end"]).any(),
        "post_period_end"
    )

    # Compile all base layers into a dictionary
    base_layers = {
        "lines": base_lines,
        "band": uncertainty_band,
        "hline": horizontal_zero,
        "vlines": vertical_lines
    }

    return base_layers


def _draw_classic_plot(plot_df: pd.DataFrame, **kwargs) -> alt.Chart:
    """Draw the classic static impact plot as in the R package.

    Args:
      plot_df: dataframe to use for plotting.
      **kwargs: Optional plot parameters. chart_width - integer for chart width in
        pixels. Default = 600. chart_height - integer for chart height in pixels.
        Default = 200. axis_title_font_size - integer for axis title font size.
        Default = 18. axis_label_font_size - integer for axis title font size.
        Default = 16. strip_title_font_size - integer for facet label font size.
        Default = 20.

    Returns:
      alt.Chart object.
    """
    layers = _create_base_layers(plot_df, **kwargs)

    # Add color spec to lines.
    color_spec = alt.Color(
        "stat_pretty:N",
        legend=alt.Legend(
            title="",
            labelFontSize=kwargs["axis_label_font_size"],
            symbolSize=10 * kwargs["axis_label_font_size"]))
    layers["lines"] = layers["lines"].encode(color=color_spec)

    # Unpack the vertical rule chart objects into a list; combine with the other
    # chart layers into a tuple that can be passed to alt.layer() to create the
    # final plot.
    vlines = list(layers["vlines"].values())
    chart_layers = tuple([layers["lines"], layers["band"], layers["hline"]] +
                         vlines)
    final_plot = alt.layer(
        *chart_layers, data=plot_df).facet(
        row=alt.Row(
            "scale_pretty:N",
            sort=["Original", "Pointwise", "Cumulative"],
            title="")).resolve_scale(y="independent").configure(
        background="white").configure_axis(
        titleFontSize=kwargs["axis_title_font_size"],
        labelFontSize=kwargs["axis_label_font_size"]
    ).configure_header(
        labelFontSize=kwargs["strip_title_font_size"])
    return final_plot


def _draw_interactive_plot(plot_df: pd.DataFrame, **kwargs) -> alt.Chart:
    """Draw interactive impact plot.

    Args:
      plot_df: dataframe to use for plotting
      **kwargs: Optional plot parameters. chart_width - integer for chart width in
        inches. Default = 15. chart_height - integer for chart height. Default =
        15. axis_title_font_size - integer for axis title font size. Default = 18.
        axis_label_font_size - integer for axis title font size. Default = 16.
        strip_title_font_size - integer for facet label font size. Default = 20.

    Returns:
      altair Chart object.
    """

    # ############################################################################
    # Create interactive selection elements.
    # ############################################################################

    # Brush for selecting a location to zoom into on x-axis.
    brush = alt.selection(type="interval", encodings=["x"])

    # Mini-chart as interactive legend to choose which stat to display.
    stat_selection = alt.selection_multi(fields=["stat_pretty"])
    selection_color = alt.condition(stat_selection,
                                    alt.Color("stat_pretty:N", legend=None),
                                    alt.value("lightgray"))
    legend = alt.Chart(plot_df).mark_point().encode(
        y=alt.Y("stat_pretty:N", axis=alt.Axis(orient="right"), title=""),
        color=selection_color).add_selection(stat_selection)

    # ############################################################################
    # Create the static top chart.
    # ############################################################################

    # The static chart is for the data on the original scale.
    static_df = plot_df.loc[plot_df["scale"] == "original"]

    # Create static layers and add the color spec to the lines layer and the
    # brush selection to the bands layer.
    static_layers = _create_base_layers(static_df, **kwargs)
    static_color_spec = alt.Color(
        "stat_pretty:N",
        legend=alt.Legend(
            title="",
            labelFontSize=kwargs["axis_label_font_size"],
            symbolSize=10 * kwargs["axis_label_font_size"]))
    static_layers["lines"] = static_layers["lines"].encode(
        color=static_color_spec)
    static_layers["band"] = static_layers["band"].add_selection(brush)

    # Combine layers of static top chart. Add the brush selection to the band
    # layer to enable selecting a date range on the x-axis for the bottom plots to
    # zoom in on (you only need to put it on one layer, so it could also have been
    # added to the lines layer).
    row_spec = alt.Row(
        "scale_pretty:N", sort=["Original", "Pointwise", "Cumulative"], title="")
    # Unpack the vertical rule chart objects into a list; combine with the other
    # chart layers into a tuple that can be passed to alt.layer to create the
    # full static plot.
    static_vlines = list(static_layers["vlines"].values())
    top_chart_layers = tuple(
        [static_layers["lines"], static_layers["band"], static_layers["hline"]] +
        static_vlines)
    top_static_plot = alt.layer(
        *top_chart_layers,
        data=static_df).facet(row=row_spec).resolve_scale(y="independent")

    # ############################################################################
    # Create the dynamic charts that will zoom upon selection on the static top
    # chart.
    # ############################################################################

    # Create dynamic layers and add interactive selections.
    dynamic_layers = _create_base_layers(plot_df, **kwargs)
    dynamic_layers["lines"] = dynamic_layers["lines"].encode(
        color=selection_color,
        x=alt.X("time", scale=alt.Scale(domain=brush), title="Time"))
    dynamic_layers["band"] = dynamic_layers["band"].encode(
        x=alt.X("time", scale=alt.Scale(domain=brush), title="Time"))

    # Add interactive selections to each of the vertical line chart objects.
    for vline_date, vline_object in dynamic_layers["vlines"].items():
        dynamic_layers["vlines"][vline_date] = vline_object.encode(
            x=alt.X(vline_date, scale=alt.Scale(domain=brush)))

    # Combine the dynamic chart layers into a tuple that can be used with
    # alt.layer() to create the full dynamic plot.
    dynamic_vlines = list(dynamic_layers["vlines"].values())
    bottom_chart_layers = tuple([
                                    dynamic_layers["lines"], dynamic_layers["band"], dynamic_layers["hline"]
                                ] + dynamic_vlines)
    bottom_dynamic_plot = alt.layer(
        *bottom_chart_layers,
        data=plot_df).facet(row=row_spec).resolve_scale(y="independent")

    # ############################################################################
    # Create the final chart.
    # ############################################################################

    # First, vertically concatenate the static and dynamic charts, then
    # horizontally concatenate with little interactive legend chart.
    final_chart = alt.vconcat(top_static_plot, bottom_dynamic_plot)
    return (final_chart | legend).configure(background="white").configure_axis(
        titleFontSize=kwargs["axis_title_font_size"],
        labelFontSize=kwargs["axis_label_font_size"]).configure_header(
        labelFontSize=kwargs["strip_title_font_size"])
