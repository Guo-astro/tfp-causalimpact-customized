# Copyright 2019-2023 The TFP CausalImpact Authors
# Copyright 2014 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This software is distributed on an "AS IS" BASIS,
# without warranties or conditions of any kind.
import arviz as az

import matplotlib.pyplot as plt
import dataclasses
import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any

from causalimpact_gibbs import posterior_processing
import causalimpact_gibbs.data as cid
from causalimpact_gibbs.indices import InputDateType, OutputDateType, OutputPeriodType

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.distributions import MultivariateNormalPrecisionFactorLinearOperator
from tensorflow_probability.python.experimental.sts_gibbs import gibbs_sampler
from tensorflow_probability.python.internal import prefer_static as ps

tfb = tfp.bijectors
tfd = tfp.distributions

TensorLike = tf.types.experimental.TensorLike
_SeedType = Union[int, Tuple[int, int], TensorLike]


@dataclasses.dataclass
class CausalImpactPosteriorSamples:
    r"""
    Posterior samples of model parameters and latent states.

    Attributes
    ----------
    observation_noise_scale : tf.Tensor
        Samples of the observation noise scale, $\sigma_{\varepsilon}$.
    level_scale : Optional[tf.Tensor]
        Samples of the local level's scale parameter.
    level : Optional[tf.Tensor]
        Samples of the local level state $\ell_t$.
    weights : Optional[tf.Tensor]
        Samples of regression coefficients $\beta$.
    seasonal_drift_scales : Optional[tf.Tensor]
        Samples of seasonal drift scales.
    seasonal_levels : Optional[tf.Tensor]
        Samples of seasonal effect states.
    """
    observation_noise_scale: tf.Tensor
    level_scale: Optional[tf.Tensor]
    level: Optional[tf.Tensor]
    weights: Optional[tf.Tensor]
    seasonal_drift_scales: Optional[tf.Tensor]
    seasonal_levels: Optional[tf.Tensor]


@dataclasses.dataclass
class CausalImpactAnalysis:
    r"""
    Results of a CausalImpact analysis.

    Attributes
    ----------
    series : pd.DataFrame
        Time-indexed DataFrame with:

        - observed: $`y_t`$
        - `posterior_mean`: $\hat{y}_t$ (posterior mean prediction)
        - `posterior_lower`, `posterior_upper`: Credible interval bounds for $\hat{y}_t$.
        - `point_effects_mean`, `point_effects_lower`, `point_effects_upper`: Mean and CI for point effects $y_t - \hat{y}_t$.
        - `cumulative_effects_mean`, `cumulative_effects_lower`, `cumulative_effects_upper`: Mean and CI for cumulative effects.

    summary : pd.DataFrame
        Summary over the post-intervention period, including:
        - Actual vs predicted outcomes
        - Absolute and relative effects
        - p-values

    posterior_samples : CausalImpactPosteriorSamples
        Posterior samples of latent variables and parameters.
    """
    series: pd.DataFrame
    summary: pd.DataFrame
    posterior_samples: CausalImpactPosteriorSamples
    convergence_diagnostics: Dict[str, Any]  # New attribute for diagnostics


@dataclasses.dataclass
class DataOptions:
    r"""
    Data configuration options.

    Attributes
    ----------
    outcome_column : Optional[str]
        Name of the column in `data` containing $`y_t`$.
    standardize_data : bool
        Whether to standardize $y_t$ and covariates.
    dtype : tf.dtypes.DType
        Data type for computations.
    """
    outcome_column: Optional[str] = None
    standardize_data: bool = True
    dtype: tf.dtypes.DType = tf.float32


@dataclasses.dataclass(frozen=True)
class Seasons:
    r"""
    Seasonal modeling options.

    Attributes
    ----------
    num_seasons : int
        Number of seasons in the cycle.
    num_steps_per_season : Union[int, Tuple[int], Tuple[Tuple[int]]]
        Steps within each season. Can be a single integer or a tuple for complex patterns.
    """
    num_seasons: int
    num_steps_per_season: Union[int, Tuple[int], Tuple[Tuple[int]]] = 1


@dataclasses.dataclass
class ModelOptions:
    r"""
    Model configuration parameters.

    Attributes
    ----------
    prior_level_sd : float
        Prior standard deviation for the local level, relative to data's std. dev.
    seasons : List[Seasons]
        List of seasonal components.
    """
    prior_level_sd: float = 0.01
    seasons: List[Seasons] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class InferenceOptions:
    r"""
    Options for posterior inference (MCMC).

    Attributes
    ----------
    num_results : int
        Number of posterior draws.
    num_warmup_steps : Optional[int]
        Number of warmup (burn-in) steps.
    """
    num_results: int = 1800
    num_warmup_steps: Optional[int] = None

    def __post_init__(self):
        # If not set, use about 1/9 of num_results as warmup
        if self.num_warmup_steps is None:
            self.num_warmup_steps = math.ceil(self.num_results / 2)


def fit_causalimpact(data: pd.DataFrame,
                     pre_period: Tuple[InputDateType, InputDateType],
                     post_period: Tuple[InputDateType, InputDateType],
                     alpha: float = 0.05,
                     seed: Optional[_SeedType] = None,
                     data_options: Optional[DataOptions] = None,
                     model_options: Optional[ModelOptions] = None,
                     inference_options: Optional[InferenceOptions] = None,
                     num_chains: int = 4,
                     **kwargs) -> CausalImpactAnalysis:
    r"""
    Fit the CausalImpact model and compute posterior effects.

    Steps:
    1. Process data, identify pre/post intervention periods.
    2. Fit an STS model using MCMC (Gibbs sampling) on pre-period data.
    3. Obtain posterior predictions $\hat{y}_t$ over the full period.
    4. Compute point effects $y_t - \hat{y}_t$ and cumulative effects.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with the outcome time series $y_t$ and optionally covariates.
    pre_period : tuple
        (start, end) of the pre-intervention period.
    post_period : tuple
        (start, end) of the post-intervention period.
    alpha : float, optional
        Significance level for credible intervals (default 0.05 for 95% CI).
    seed : int, tuple, or TensorLike, optional
        Random seed for reproducibility.
    data_options : DataOptions, optional
        Configuration for data handling.
    model_options : ModelOptions, optional
        Model hyperparameters.
    inference_options : InferenceOptions, optional
        MCMC sampling configurations.
    **kwargs
        Experimental arguments, if any.

    Returns
    -------
    CausalImpactAnalysis
        Analysis results, including posterior samples and effect estimates.
    """
    tf_log_level = tf.get_logger().level
    tf.get_logger().setLevel(logging.ERROR)
    try:
        data_options = data_options if data_options is not None else DataOptions()
        model_options = model_options if model_options is not None else ModelOptions()
        inference_options = inference_options if inference_options is not None else InferenceOptions()

        experimental_model = kwargs.pop("experimental_model", None)
        experimental_tf_function_cache_key_addition = kwargs.pop("experimental_tf_function_cache_key_addition", 0)
        if kwargs:
            raise TypeError(f"Received unknown {kwargs=}")

        ci_data = cid.CausalImpactData(
            data=data,
            pre_intervention_period=pre_period,
            post_intervention_period=post_period,
            target_col_name=data_options.outcome_column,
            standardize_data=data_options.standardize_data,
            dtype=data_options.dtype)
        # Initialize lists to store samples from all chains
        all_posterior_samples = []
        all_posterior_means = []
        all_posterior_trajectories = []
        # Handle seed
        if isinstance(seed, int):
            if num_chains == 1:
                seeds = (seed,)
            else:
                # Generate deterministic seeds from the base seed
                seeds = tuple(tfp.random.sanitize_seed((seed, chain)) for chain in range(num_chains))
        elif isinstance(seed, tuple):
            raise ValueError(f"Seed tuple not supported now, got {seed}.")
        elif seed is None:
            # Use default seeding if no seed is provided
            seeds = tuple(tfp.random.sanitize_seed(None) for _ in range(num_chains))
        else:
            raise TypeError(f"Seed must be an int, tuple of ints, or None, got {type(seed)}.")
        # Run Gibbs sampling for each chain
        for chain in range(num_chains):
            print(f"Running chain {chain + 1}/{num_chains}...")
            posterior_samples, posterior_means, posterior_trajectories = _train_causalimpact_sts(
                ci_data=ci_data,
                prior_level_sd=model_options.prior_level_sd,
                seed=seeds[chain],
                num_results=inference_options.num_results,
                num_warmup_steps=inference_options.num_warmup_steps,
                model=experimental_model,
                dtype=data_options.dtype,
                seasons=model_options.seasons,
                experimental_tf_function_cache_key_addition=experimental_tf_function_cache_key_addition
            )
            all_posterior_samples.append(posterior_samples)
            all_posterior_means.append(posterior_means)
            all_posterior_trajectories.append(posterior_trajectories)
            # Combine samples from all chains
        combined_posterior_samples = _combine_posterior_samples(all_posterior_samples)
        combined_posterior_means = tf.concat(all_posterior_means, axis=0)
        combined_posterior_trajectories = tf.concat(all_posterior_trajectories, axis=0)

        # Compute convergence diagnostics
        convergence_diagnostics = _compute_convergence_diagnostics(combined_posterior_samples, num_chains)

        series, summary = _compute_impact(
            posterior_means=combined_posterior_means,
            posterior_trajectories=combined_posterior_trajectories,
            ci_data=ci_data,
            num_chains=num_chains,
            alpha=alpha)

        if combined_posterior_samples.seasonal_levels.shape[-1] > 0:
            seasonal_levels = []
            idx = 0
            for season in model_options.seasons:
                seasonal_levels.append(combined_posterior_samples.seasonal_levels[..., idx])
                idx += season.num_seasons - 1
            seasonal_levels = tf.stack(seasonal_levels, axis=-1)
        else:
            seasonal_levels = combined_posterior_samples.seasonal_levels

        ci_posterior_samples = CausalImpactPosteriorSamples(
            observation_noise_scale=combined_posterior_samples.observation_noise_scale,
            level_scale=combined_posterior_samples.level_scale,
            level=combined_posterior_samples.level,
            weights=(combined_posterior_samples.weights
                     if combined_posterior_samples.weights.shape[1] > 0 else None),
            seasonal_drift_scales=(
                combined_posterior_samples.seasonal_drift_scales
                if combined_posterior_samples.seasonal_drift_scales.shape[-1] > 0 else None),
            seasonal_levels=seasonal_levels
        )

        return CausalImpactAnalysis(series, summary, ci_posterior_samples, convergence_diagnostics)
    finally:
        tf.get_logger().setLevel(tf_log_level)


def _combine_posterior_samples(
        posterior_samples_list: List[gibbs_sampler.GibbsSamplerState]) -> gibbs_sampler.GibbsSamplerState:
    """
    Combine posterior samples from multiple chains.

    Parameters
    ----------
    posterior_samples_list : List[gibbs_sampler.GibbsSamplerState]
        List of posterior samples from each chain.

    Returns
    -------
    gibbs_sampler.GibbsSamplerState
        Combined posterior samples.
    """
    # Assuming all chains have the same structure
    combined_observation_noise_scale = tf.concat([s.observation_noise_scale for s in posterior_samples_list], axis=0)
    combined_level_scale = tf.concat([s.level_scale for s in posterior_samples_list], axis=0) if posterior_samples_list[
                                                                                                     0].level_scale is not None else None
    combined_level = tf.concat([s.level for s in posterior_samples_list], axis=0) if posterior_samples_list[
                                                                                         0].level is not None else None
    combined_weights = tf.concat([s.weights for s in posterior_samples_list], axis=0) if posterior_samples_list[
                                                                                             0].weights is not None else None
    combined_seasonal_drift_scales = tf.concat([s.seasonal_drift_scales for s in posterior_samples_list], axis=0) if \
        posterior_samples_list[0].seasonal_drift_scales is not None else None
    combined_seasonal_levels = tf.concat([s.seasonal_levels for s in posterior_samples_list], axis=0) if \
        posterior_samples_list[0].seasonal_levels is not None else None

    return gibbs_sampler.GibbsSamplerState(
        seed=123,
        observation_noise_scale=combined_observation_noise_scale,
        level_scale=combined_level_scale,
        level=combined_level,
        weights=combined_weights,
        seasonal_drift_scales=combined_seasonal_drift_scales,
        seasonal_levels=combined_seasonal_levels
    )


def _compute_convergence_diagnostics(posterior_samples: gibbs_sampler.GibbsSamplerState,
                                     num_chains: int) -> Dict[str, Any]:
    """
    Compute convergence diagnostics using ArviZ.
    """

    # Convert posterior samples to ArviZ InferenceData
    data_dict = {}
    chain_size = posterior_samples.observation_noise_scale.shape[0] // num_chains

    # Debugging: Print shapes to verify data
    print(f"Observation noise scale shape: {posterior_samples.observation_noise_scale.shape}")
    data_dict["observation_noise_scale"] = posterior_samples.observation_noise_scale.numpy().reshape(num_chains,
                                                                                                     chain_size)

    if posterior_samples.level_scale is not None:
        print(f"Level scale shape: {posterior_samples.level_scale.shape}")
        data_dict["level_scale"] = posterior_samples.level_scale.numpy().reshape(num_chains, chain_size)
    if posterior_samples.level is not None:
        print(f"Level shape: {posterior_samples.level.shape}")
        data_dict["level"] = posterior_samples.level.numpy().reshape(num_chains, chain_size, -1)
    if posterior_samples.weights is not None:
        print(f"Weights shape: {posterior_samples.weights.shape}")
        data_dict["weights"] = posterior_samples.weights.numpy().reshape(num_chains, chain_size, -1)
    if posterior_samples.seasonal_drift_scales is not None:
        print(f"Seasonal drift scales shape: {posterior_samples.seasonal_drift_scales.shape}")
        data_dict["seasonal_drift_scales"] = posterior_samples.seasonal_drift_scales.numpy().reshape(num_chains,
                                                                                                     chain_size, -1)
    if posterior_samples.seasonal_levels is not None:
        print(f"Seasonal levels shape: {posterior_samples.seasonal_levels.shape}")
        data_dict["seasonal_levels"] = posterior_samples.seasonal_levels.numpy().reshape(num_chains, chain_size, -1)

    print(f"Data dict keys: {data_dict.keys()}")

    if not data_dict:
        raise ValueError("No posterior samples available for convergence diagnostics.")

    return data_dict


@tf.function(autograph=False, jit_compile=False)
def _run_gibbs_sampler(
        sts_model: Optional[tfp.sts.StructuralTimeSeries],
        outcome_ts: TensorLike,
        outcome_sd: TensorLike,
        design_matrix: Optional[TensorLike],
        num_results: int,
        num_warmup_steps: int,
        observation_noise_scale: TensorLike,
        level_scale: TensorLike,
        seasonal_drift_scales: TensorLike,
        weights: TensorLike,
        level: TensorLike,
        slope: TensorLike,
        seed: TensorLike,
        dtype,
        seasons: List[Seasons],
        experimental_tf_function_cache_key_addition: int):
    r"""
    Run Gibbs sampling for STS parameters and states.

    Given a structural time series model and observed data (with possible missing values),
    this function draws samples from the posterior distribution over parameters and latent states
    using Gibbs sampling.

    Parameters
    ----------
    sts_model : tfp.sts.StructuralTimeSeries, optional
        The STS model. If None, a default model is constructed.
    outcome_ts : TensorLike
        Time series of $y_t$, possibly with NaNs for post-intervention.
    outcome_sd : TensorLike
        Standard deviation of the pre-intervention data, used to scale priors.
    design_matrix : Optional[TensorLike]
        Covariates, if any.
    num_results : int
        Number of MCMC samples to draw.
    num_warmup_steps : int
        Warmup (burn-in) steps before collecting samples.
    observation_noise_scale : TensorLike
        Initial guess for observation noise scale $\sigma_{\varepsilon}$.
    level_scale : TensorLike
        Prior scale for the local level component.
    seasonal_drift_scales : TensorLike
        Drift scales for seasonal effects.
    weights : TensorLike
        Initial regression weights (if covariates are included).
    level : TensorLike
        Initial level state.
    slope : TensorLike
        Initial slope state (if any, else zero).
    seed : TensorLike
        PRNG seed.
    dtype :
        Data type for computations.
    seasons : List[Seasons]
        Seasonal configuration.
    experimental_tf_function_cache_key_addition : int
        Experimental parameter to control tracing.

    Returns
    -------
    posterior_samples : gibbs_sampler.GibbsSamplerState
        Samples of parameters and states.
    posterior_means : tf.Tensor
        Posterior mean predictions $\hat{y}_t$.
    posterior_trajectories : tf.Tensor
        Posterior predictive samples for $\hat{y}_t$.
    """
    if not sts_model:
        sts_model = _build_default_gibbs_model(
            design_matrix=design_matrix,
            outcome_ts=outcome_ts,
            level_scale=level_scale,
            outcome_sd=outcome_sd,
            dtype=dtype,
            seasons=seasons)
    tf.random.set_seed(seed)
    sample_seed, forecast_seed = tfp.random.split_seed(seed)
    posterior_samples = gibbs_sampler.fit_with_gibbs_sampling(
        sts_model,
        outcome_ts,
        num_results=num_results,
        num_warmup_steps=num_warmup_steps,
        initial_state=gibbs_sampler.GibbsSamplerState(
            observation_noise_scale=observation_noise_scale,
            level_scale=level_scale,
            slope_scale=tf.zeros([], dtype=dtype),
            weights=weights,
            level=level,
            slope=slope,
            seed=None,
            seasonal_drift_scales=seasonal_drift_scales,
            seasonal_levels=tf.zeros(
                shape=gibbs_sampler.get_seasonal_latents_shape(
                    outcome_ts.time_series, sts_model),
                dtype=dtype)),
        default_pseudo_observations=tf.ones([], dtype=dtype) * 0.01,
        seed=sample_seed,
        experimental_use_dynamic_cholesky=True,
        experimental_use_weight_adjustment=True)

    posterior_means, posterior_trajectories = _get_posterior_means_and_trajectories(
        sts_model=sts_model,
        posterior_samples=posterior_samples,
        seed=forecast_seed)

    return posterior_samples, posterior_means, posterior_trajectories


def _build_default_gibbs_model(
        design_matrix: Optional[tf.Tensor],
        outcome_ts: tfp.sts.MaskedTimeSeries,
        level_scale: tf.Tensor,
        outcome_sd: tf.Tensor,
        dtype,
        seasons: List[Seasons],
):
    """
    Construct a default STS model for Gibbs sampling.

    Includes:
    - A local level component with an InverseGamma prior on the level variance.
    - Optional regression using covariates (if provided).
    - Optional seasonal components.

    Parameters
    ----------
    design_matrix : Optional[tf.Tensor]
        Covariates for regression.
    outcome_ts : tfp.sts.MaskedTimeSeries
        Observed series with possible NaNs.
    level_scale : tf.Tensor
        Scale for the local level prior.
    outcome_sd : tf.Tensor
        Standard deviation of pre-period data.
    dtype :
        Data type.
    seasons : List[Seasons]
        Seasonal configurations.

    Returns
    -------
    sts_model : tfp.sts.StructuralTimeSeries
        The constructed STS model suitable for Gibbs sampling.
    """
    local_level_prior_sample_size = tf.constant(32., dtype=dtype)
    level_concentration = tf.cast(local_level_prior_sample_size / 2., dtype=dtype)
    level_variance_prior_scale = level_scale * level_scale * (local_level_prior_sample_size / 2.)
    level_variance_prior = tfd.InverseGamma(
        concentration=level_concentration, scale=level_variance_prior_scale)
    level_variance_prior.upper_bound = outcome_sd

    if design_matrix is not None:
        observation_noise_variance_prior = tfd.InverseGamma(
            concentration=tf.constant(25., dtype=dtype),
            scale=tf.math.square(outcome_sd) * tf.constant(5., dtype=dtype))
    else:
        observation_noise_variance_prior = tfd.InverseGamma(
            concentration=tf.constant(0.005, dtype=dtype),
            scale=tf.math.square(outcome_sd) * tf.constant(0.005, dtype=dtype))
    observation_noise_variance_prior.upper_bound = outcome_sd * tf.constant(1.2, dtype=dtype)

    if design_matrix is not None:
        design_shape = ps.shape(design_matrix)
        num_outputs = design_shape[-2]
        num_dimensions = design_shape[-1]
        sparse_weights_nonzero_prob = tf.minimum(
            tf.constant(1., dtype=dtype), 3. / num_dimensions)
        x_transpose_x = tf.matmul(design_matrix, design_matrix, transpose_a=True)
        weights_prior_precision = 0.01 * tf.linalg.set_diag(
            0.5 * x_transpose_x, tf.linalg.diag_part(x_transpose_x)) / num_outputs
        precision_factor = tf.linalg.cholesky(weights_prior_precision)
        weights_prior = MultivariateNormalPrecisionFactorLinearOperator(
            precision_factor=tf.linalg.LinearOperatorFullMatrix(precision_factor),
            precision=tf.linalg.LinearOperatorFullMatrix(weights_prior_precision))
    else:
        sparse_weights_nonzero_prob = None
        weights_prior = None

    initial_level_prior = tfd.Normal(
        loc=tf.cast(outcome_ts.time_series[..., 0], dtype=dtype),
        scale=outcome_sd)

    seasonal_components = []
    seasonal_variance_prior = tfd.InverseGamma(
        concentration=0.005, scale=5e-7 * tf.square(outcome_sd))
    seasonal_variance_prior.upper_bound = outcome_sd
    for seasonal_options in seasons:
        seasonal_components.append(
            tfp.sts.Seasonal(
                num_seasons=seasonal_options.num_seasons,
                num_steps_per_season=np.array(seasonal_options.num_steps_per_season),
                allow_drift=True,
                constrain_mean_effect_to_zero=True,
                drift_scale_prior=tfd.TransformedDistribution(
                    bijector=tfb.Invert(tfb.Square()),
                    distribution=seasonal_variance_prior),
                initial_effect_prior=tfd.Normal(loc=0., scale=outcome_sd)))

    return gibbs_sampler.build_model_for_gibbs_fitting(
        outcome_ts,
        design_matrix=design_matrix,
        weights_prior=weights_prior,
        level_variance_prior=level_variance_prior,
        slope_variance_prior=None,
        observation_noise_variance_prior=observation_noise_variance_prior,
        initial_level_prior=initial_level_prior,
        sparse_weights_nonzero_prob=sparse_weights_nonzero_prob,
        seasonal_components=seasonal_components)


def _train_causalimpact_sts(
        *,
        ci_data: cid.CausalImpactData,
        prior_level_sd,
        seed: _SeedType,
        num_results: int,
        num_warmup_steps: int,
        model: Optional[tfp.sts.StructuralTimeSeries] = None,
        dtype,
        seasons: List[Seasons],
        experimental_tf_function_cache_key_addition: int = 0,
) -> Tuple[gibbs_sampler.GibbsSamplerState, TensorLike, TensorLike]:
    r"""
    Train the STS model via Gibbs sampling and return posterior predictions.

    Steps:
    - Extend $y_t$ into the post-period with NaNs.
    - Run Gibbs sampling to get posterior samples.
    - Return posterior samples and $\hat{y}_t$ predictions.

    Parameters
    ----------
    ci_data : cid.CausalImpactData
        Processed data with pre/post intervention periods.
    prior_level_sd : float
        Prior level standard deviation (scaled).
    seed : _SeedType
        Seed for reproducibility.
    num_results : int
        Number of posterior draws.
    num_warmup_steps : int
        Warmup steps (burn-in).
    model : Optional[tfp.sts.StructuralTimeSeries]
        Custom STS model, if any.
    dtype :
        Data type.
    seasons : List[Seasons]
        Seasonal configurations.
    experimental_tf_function_cache_key_addition : int
        Experimental parameter for tracing.

    Returns
    -------
    posterior_samples : gibbs_sampler.GibbsSamplerState
        Posterior samples of parameters and states.
    y_hat_means : tf.Tensor
        Posterior mean predictions $\hat{y}_t$.
    y_hat_samples : tf.Tensor
        Posterior predictive samples of $\hat{y}_t$.
    """
    if isinstance(seed, int):
        seed = (0, seed)

    X = (tf.convert_to_tensor(ci_data.normalized_whole_period_features.values, dtype=dtype)
         if ci_data.normalized_whole_period_features is not None else None)

    post_len = ci_data.normalized_after_pre_intervention_data.shape[0]
    y_pre = ci_data.pre_intervention_target_ts.time_series
    y_extended = tf.concat([y_pre, tf.fill([post_len], tf.constant(float("nan"), dtype=dtype))], axis=0)
    is_missing_extended = tf.concat([ci_data.pre_intervention_target_ts.is_missing,
                                     tf.fill([post_len], True)], axis=0)
    extended_target_ts = tfp.sts.MaskedTimeSeries(
        time_series=y_extended, is_missing=is_missing_extended)

    y_pre_std = tf.convert_to_tensor(
        np.nanstd(ci_data.pre_intervention_target_ts.time_series, ddof=1), dtype=dtype)

    r2 = 0.8
    if X is not None:
        observation_noise_scale = tf.cast(tf.math.sqrt(1 - r2), dtype=dtype) * y_pre_std
    else:
        observation_noise_scale = y_pre_std

    level_scale = tf.ones([], dtype=dtype) * prior_level_sd * y_pre_std
    seasonal_drift_scales = 0.01 * y_pre_std * tf.ones(shape=[len(seasons)], dtype=dtype)

    if ci_data.normalized_whole_period_features is None:
        weights_init = tf.zeros([0], dtype=dtype)
    else:
        weights_init = tf.zeros(ci_data.normalized_whole_period_features.shape[-1:], dtype=dtype)

    level_init = tf.zeros_like(extended_target_ts.time_series)
    slope_init = tf.zeros_like(extended_target_ts.time_series)

    posterior_samples, y_hat_means, y_hat_samples = _run_gibbs_sampler(
        sts_model=model,
        outcome_ts=extended_target_ts,
        outcome_sd=y_pre_std,
        design_matrix=X,
        num_results=num_results,
        num_warmup_steps=num_warmup_steps,
        observation_noise_scale=observation_noise_scale,
        level_scale=level_scale,
        seasonal_drift_scales=seasonal_drift_scales,
        weights=weights_init,
        level=level_init,
        slope=slope_init,
        seed=seed,
        dtype=dtype,
        seasons=seasons,
        experimental_tf_function_cache_key_addition=experimental_tf_function_cache_key_addition
    )

    return posterior_samples, y_hat_means, y_hat_samples


def _get_posterior_means_and_trajectories(sts_model, posterior_samples, seed):
    r"""
    Extract posterior mean and trajectory samples of $\hat{y}_t$.

    Parameters
    ----------
    sts_model : tfp.sts.StructuralTimeSeries
        The STS model used.
    posterior_samples : gibbs_sampler.GibbsSamplerState
        Posterior samples of parameters and states.
    seed : int or tuple
        Seed for sampling predictive trajectories.

    Returns
    -------
    y_hat_mean : tf.Tensor
        Mean $\hat{y}_t$ predictions.
    y_hat_samples : tf.Tensor
        Sampled trajectories of $\hat{y}_t$.
    """
    predictive_distributions = gibbs_sampler.one_step_predictive(
        sts_model,
        posterior_samples,
        thin_every=1,
        use_zero_step_prediction=True)

    y_hat_mean = predictive_distributions.mean()
    y_hat_samples = tf.transpose(predictive_distributions.components_distribution.sample(seed=seed))
    return y_hat_mean, y_hat_samples


def _compute_impact(
        posterior_means,
        posterior_trajectories,
        ci_data: cid.CausalImpactData,
        num_chains: int,
        alpha: float = 0.05,

) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Compute time-series effects and summary statistics.

    - Derive credible intervals for $\hat{y}_t$.
    - Compute point effects $y_t - \hat{y}_t$.
    - Compute cumulative effects.
    - Summarize results.

    Parameters
    ----------
    posterior_means : tf.Tensor
        Posterior mean predictions of $\hat{y}_t$.
    posterior_trajectories : tf.Tensor
        Posterior predictive samples of $\hat{y}_t$.
    ci_data : cid.CausalImpactData
        Data structure with pre/post intervention info.
    alpha : float
        Significance level for intervals.

    Returns
    -------
    series : pd.DataFrame
        Detailed time series with observed, predicted, and effect columns.
    summary : pd.DataFrame
        Aggregate summary over the post-intervention period.
    """
    if not 0 < alpha < 1:
        raise ValueError("`alpha` must be between 0 and 1.")

    observed_pre = ci_data.pre_intervention_data[ci_data.target_col]
    observed_post = ci_data.after_pre_intervention_data[ci_data.target_col]
    observed_post = observed_post.loc[(observed_post.index >= ci_data.post_intervention_period[0]) &
                                      (observed_post.index <= ci_data.post_intervention_period[1])]
    observed_full = pd.concat([observed_pre, observed_post], axis=0)

    quantiles = (alpha / 2.0, 1.0 - (alpha / 2.0))

    posterior_trajectories, posterior_trajectory_summary = (
        _sample_posterior_predictive(
            posterior_means=posterior_means,
            posterior_trajectories=posterior_trajectories,
            ci_data=ci_data,
            quantiles=quantiles, num_chains=num_chains))

    trajectory_dict = _compute_impact_trajectories(
        posterior_trajectories,
        observed_full,
        treatment_start=ci_data.post_intervention_period[0])

    series = _compute_impact_estimates(
        posterior_trajectory_summary=posterior_trajectory_summary,
        trajectory_dict=trajectory_dict,
        observed_ts_full=observed_full,
        ci_data=ci_data,
        quantiles=quantiles)

    summary = _compute_summary(
        posterior_trajectory_summary=posterior_trajectory_summary,
        trajectory_dict=trajectory_dict,
        observed_ts_post=observed_post,
        post_period=ci_data.post_intervention_period,
        quantiles=quantiles,
        alpha=alpha)

    return series, summary


def _sample_posterior_predictive(
        posterior_means: tf.Tensor,
        posterior_trajectories: tf.Tensor,
        ci_data: cid.CausalImpactData,
        quantiles: Tuple[float, float],
        num_chains: int  # Add this parameter
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r"""
    Package posterior predictions into DataFrames with mean and quantiles.

    Parameters
    ----------
    posterior_means : tf.Tensor
        Posterior mean $\hat{y}_t$.
    posterior_trajectories : tf.Tensor
        Posterior samples of $\hat{y}_t$.
    ci_data : cid.CausalImpactData
        Data structure with full data info.
    quantiles : tuple
        Quantiles for intervals (e.g., (0.025, 0.975)).
    num_chains : int
        Number of MCMC chains used in sampling.

    Returns
    -------
    posterior_trajectories : pd.DataFrame
        DataFrame of sampled trajectories.
    posterior_trajectory_summary : pd.DataFrame
        DataFrame with posterior mean and quantile columns.
    """
    # Calculate samples per chain
    total_samples = posterior_means.shape[0]
    samples_per_chain = total_samples // num_chains

    if total_samples % num_chains != 0:
        raise ValueError(f"Total posterior_means samples ({total_samples}) not divisible by num_chains ({num_chains}).")

    # Reshape to (num_chains, samples_per_chain, num_features)
    # Assuming posterior_means shape is (num_chains * samples_per_chain, num_features)
    reshaped_posterior_means = tf.reshape(posterior_means, (num_chains, samples_per_chain, -1))

    # Average across chains to get (samples_per_chain, num_features)
    averaged_posterior_means = tf.reduce_mean(reshaped_posterior_means, axis=0)

    # Transpose to (num_features, samples_per_chain) if necessary
    # Assuming num_features=1 for posterior_mean
    averaged_posterior_means = tf.transpose(averaged_posterior_means)  # Shape: (num_features, samples_per_chain)

    # Convert to DataFrame
    posterior_means_df = posterior_processing.process_posterior_quantities(
        ci_data, averaged_posterior_means, ["posterior_mean"]
    )

    # Handle posterior trajectories by packaging them correctly
    posterior_trajectories_df = _package_posterior_trajectories(
        posterior_trajectories, ci_data, num_chains
    )

    # Calculate quantiles
    posterior_quantiles = posterior_processing.calculate_trajectory_quantiles(
        posterior_trajectories_df, "posterior", quantiles
    )

    # Join posterior means with quantiles
    posterior_trajectory_summary = posterior_means_df.join(posterior_quantiles)

    return posterior_trajectories_df, posterior_trajectory_summary


def _package_posterior_trajectories(
        posterior_trajectories: tf.Tensor,
        ci_data: cid.CausalImpactData,
        num_chains: int  # Add this parameter to handle multiple chains
) -> pd.DataFrame:
    r"""
    Convert sampled $\hat{y}_t$ trajectories into a DataFrame.

    Columns are `sample_1, sample_2, ...`

    Parameters
    ----------
    posterior_trajectories : tf.Tensor
        Posterior samples of predictions.
    ci_data : cid.CausalImpactData
        Data for indexing.
    num_chains : int
        Number of MCMC chains used in sampling.

    Returns
    -------
    pd.DataFrame
        DataFrame with each column a sample trajectory.
    """
    total_samples = posterior_trajectories.shape[0]
    samples_per_chain = total_samples // num_chains

    if total_samples % num_chains != 0:
        raise ValueError(
            f"Total posterior_trajectories samples ({total_samples}) not divisible by num_chains ({num_chains}).")

    # Reshape to (num_chains, samples_per_chain, num_time_points)
    reshaped_posterior_trajectories = tf.reshape(posterior_trajectories, (num_chains, samples_per_chain, -1))

    # Flatten all chains and samples into a single axis (num_chains * samples_per_chain, num_time_points)
    flattened_posterior_trajectories = tf.reshape(reshaped_posterior_trajectories,
                                                  (-1, reshaped_posterior_trajectories.shape[2]))

    # Transpose to (num_time_points, num_samples)
    transposed_posterior_trajectories = tf.transpose(flattened_posterior_trajectories)

    # Create column names
    num_samples = transposed_posterior_trajectories.shape[1]
    col_names = [f"sample_{i + 1}" for i in range(num_samples)]

    # Convert to DataFrame
    return posterior_processing.process_posterior_quantities(
        ci_data, transposed_posterior_trajectories.numpy(), col_names
    )


def _compute_impact_trajectories(
        posterior_trajectories: pd.DataFrame, y_full: pd.Series,
        treatment_start: OutputDateType) -> Dict[str, pd.DataFrame]:
    r"""
    Compute point and cumulative effect trajectories.

    Point effects:
    $$y_t - \hat{y}_{t, \text{sample}}$$

    Cumulative effects:
    $$\sum_{\tau=T_{\text{start}}}^{t} (y_\tau - \hat{y}_{\tau,\text{sample}})$$

    Parameters
    ----------
    posterior_trajectories : pd.DataFrame
        DataFrame of $\hat{y}_t$ samples.
    y_full : pd.Series
        Full observed data series.
    treatment_start : OutputDateType
        Start of the post-intervention period.

    Returns
    -------
    dict of pd.DataFrame
        Contains "predictions", "point_effects", "cumulative_effects".
    """
    point_effects = posterior_trajectories.sub(y_full, axis=0).mul(-1)
    cum_base = point_effects.copy()
    cum_base.loc[cum_base.index < treatment_start] = 0
    cumulative_effects = cum_base.cumsum(axis=0)

    return {
        "predictions": posterior_trajectories,
        "point_effects": point_effects,
        "cumulative_effects": cumulative_effects
    }


def _compute_impact_estimates(posterior_trajectory_summary: pd.DataFrame,
                              trajectory_dict: Dict[str, pd.DataFrame],
                              observed_ts_full: pd.Series,
                              ci_data: cid.CausalImpactData,
                              quantiles: Tuple[float, float]) -> pd.DataFrame:
    r"""
    Construct a time series DataFrame with effects and intervals.

    - point_effects_mean: $E[y_t - \hat{y}_t]$
    - cumulative_effects_mean: $\sum_{\tau=T_{\text{start}}}^{t} (y_\tau - \hat{y}_\tau)$

    Parameters
    ----------
    posterior_trajectory_summary : pd.DataFrame
        Summary with posterior_mean and intervals of $\hat{y}_t$.
    trajectory_dict : dict
        Dict with "point_effects" and "cumulative_effects".
    observed_ts_full : pd.Series
        Full observed series (pre + post).
    ci_data : cid.CausalImpactData
        Data info.
    quantiles : tuple
        Interval quantiles.

    Returns
    -------
    pd.DataFrame
        DataFrame with observed, predicted, effects, and intervals.
    """
    point_effects_mean = (observed_ts_full - posterior_trajectory_summary["posterior_mean"])
    point_effects_mean = point_effects_mean.to_frame(name="point_effects_mean")

    zero_inds = point_effects_mean.index < ci_data.post_intervention_period[0]
    cum_effects_mean_base = point_effects_mean.copy()
    cum_effects_mean_base.loc[zero_inds] = 0
    cum_effects_mean = cum_effects_mean_base.cumsum()
    cum_effects_mean.columns = ["cumulative_effects_mean"]

    point_effects_quantiles = posterior_processing.calculate_trajectory_quantiles(
        trajectory_dict["point_effects"], "point_effects", quantiles)
    cum_effects_quantiles = posterior_processing.calculate_trajectory_quantiles(
        trajectory_dict["cumulative_effects"], "cumulative_effects", quantiles)

    impact_estimates = pd.concat([
        observed_ts_full.to_frame(name="observed"),
        posterior_trajectory_summary,
        point_effects_mean,
        point_effects_quantiles,
        cum_effects_mean,
        cum_effects_quantiles
    ], axis=1)

    impact_estimates.loc[
        ((impact_estimates.index > ci_data.pre_intervention_period[1]) &
         (impact_estimates.index < ci_data.post_intervention_period[0])) |
        (impact_estimates.index > ci_data.post_intervention_period[1]),
        impact_estimates.columns.difference(
            ["observed", "posterior_mean", "posterior_lower", "posterior_upper"]
        )] = np.nan

    impact_estimates.loc[
        np.isnan(impact_estimates["observed"]),
        impact_estimates.columns.difference(
            ["observed", "posterior_mean", "posterior_lower", "posterior_upper"]
        )] = np.nan

    impact_estimates = impact_estimates.reindex(ci_data.data.index, copy=False, fill_value=np.nan)
    impact_estimates["observed"] = ci_data.data[ci_data.target_col]

    impact_estimates["pre_period_start"] = ci_data.pre_intervention_period[0]
    impact_estimates["pre_period_end"] = ci_data.pre_intervention_period[1]
    impact_estimates["post_period_start"] = ci_data.post_intervention_period[0]
    impact_estimates["post_period_end"] = ci_data.post_intervention_period[1]

    return impact_estimates


def _compute_summary(posterior_trajectory_summary: pd.DataFrame,
                     trajectory_dict: Dict[str, pd.DataFrame],
                     observed_ts_post: pd.Series, post_period: OutputPeriodType,
                     quantiles: Tuple[float, float], alpha: float) -> pd.DataFrame:
    r"""
    Compute summary statistics over the post-period:

    - Average and cumulative predicted $\hat{y}_t$
    - Absolute effects $y_t - \hat{y}_t$
    - Relative effects $(y_t/\hat{y}_t - 1)$
    - p-value for extremeness of observed outcome

    Parameters
    ----------
    posterior_trajectory_summary : pd.DataFrame
        Posterior mean and intervals of $\hat{y}_t$.
    trajectory_dict : dict
        Contains trajectories for "point_effects" and "cumulative_effects".
    observed_ts_post : pd.Series
        Observed data in the post-period.
    post_period : tuple
        (start, end) of the post-intervention period.
    quantiles : tuple
        Interval quantiles (e.g., (0.025, 0.975)).
    alpha : float
        Significance level.

    Returns
    -------
    pd.DataFrame
        Summary of actual, predicted, and effect statistics over post-period.
    """
    posterior_mean = posterior_trajectory_summary.loc[
        (posterior_trajectory_summary.index >= post_period[0]) &
        (posterior_trajectory_summary.index <= post_period[1]), "posterior_mean"]
    trajectory_dict = {
        k: v.loc[(v.index >= post_period[0]) & (v.index <= post_period[1])]
        for k, v in trajectory_dict.items()
    }

    avg_pred = posterior_mean.mean()
    cum_pred = posterior_mean.sum()

    pred_mean_samples = trajectory_dict["predictions"].mean(axis=0)
    pred_sum_samples = trajectory_dict["predictions"].sum(axis=0)
    avg_pred_lower, avg_pred_upper = pred_mean_samples.quantile(quantiles)
    cum_pred_lower, cum_pred_upper = pred_sum_samples.quantile(quantiles)

    avg_effect = observed_ts_post.mean() - avg_pred
    avg_eff_samples = trajectory_dict["point_effects"].mean(axis=0)
    avg_eff_lower, avg_eff_upper = avg_eff_samples.quantile(quantiles)

    cum_effect = observed_ts_post.sum() - cum_pred
    cum_eff_samples = trajectory_dict["point_effects"].sum(axis=0)
    cum_eff_lower, cum_eff_upper = cum_eff_samples.quantile(quantiles)

    rel_eff_samples = (observed_ts_post.sum() / pred_sum_samples - 1.)
    rel_eff_lower, rel_eff_upper = rel_eff_samples.quantile(quantiles)

    summary_dict = {
        "actual": {"average": observed_ts_post.mean(), "cumulative": observed_ts_post.sum()},
        "predicted": {"average": avg_pred, "cumulative": cum_pred},
        "predicted_lower": {"average": avg_pred_lower, "cumulative": cum_pred_lower},
        "predicted_upper": {"average": avg_pred_upper, "cumulative": cum_pred_upper},
        "predicted_sd": {"average": pred_mean_samples.std(), "cumulative": pred_sum_samples.std()},
        "abs_effect": {"average": avg_effect, "cumulative": cum_effect},
        "abs_effect_lower": {"average": avg_eff_lower, "cumulative": cum_eff_lower},
        "abs_effect_upper": {"average": avg_eff_upper, "cumulative": cum_eff_upper},
        "abs_effect_sd": {"average": avg_eff_samples.std(), "cumulative": cum_eff_samples.std()},
        "rel_effect": {"average": rel_eff_samples.mean(), "cumulative": rel_eff_samples.mean()},
        "rel_effect_lower": {"average": rel_eff_lower, "cumulative": rel_eff_lower},
        "rel_effect_upper": {"average": rel_eff_upper, "cumulative": rel_eff_upper},
        "rel_effect_sd": {"average": rel_eff_samples.std(), "cumulative": rel_eff_samples.std()}
    }
    summary_df = pd.DataFrame(summary_dict)

    observed_cumulative_outcome = observed_ts_post.sum()
    sampled_cumulative_outcomes = pd.concat([pred_sum_samples, pd.Series(observed_cumulative_outcome)], axis=0)
    prop_obs_less = (observed_cumulative_outcome <= sampled_cumulative_outcomes).mean()
    prop_obs_greater = (observed_cumulative_outcome >= sampled_cumulative_outcomes).mean()
    p_value = min(prop_obs_less, prop_obs_greater)
    summary_df["p_value"] = p_value
    summary_df["alpha"] = alpha

    return summary_df
