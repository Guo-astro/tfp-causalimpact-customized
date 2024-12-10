import unittest
import logging
from typing import Optional
from jinja2 import Environment, Template
from datetime import datetime
import numpy as np
import pandas as pd
import os

# Import your CausalImpactAnalysis class
from causalimpact_gibbs.causalimpact_lib import CausalImpactAnalysis

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Jinja2 Templates with corrected filters
summary_text = """
{% macro CI(alpha) %}{{ (((1 - alpha) * 100) | string).rstrip('0').rstrip('.') }}% CI{% endmacro -%}
{% macro add_remaining_spaces(n) %}{{ ' ' * max(19 - n, 0) }}{% endmacro -%}
Posterior Inference {CausalImpact}
                          Average            {% if summary.cumulative %}Cumulative{% else %}          {% endif %}
Actual                    {{ summary.average.actual | round(3) }}{{ add_remaining_spaces(summary.average.actual_length | default(0)) }}{{ summary.cumulative.actual | round(3) if summary.cumulative else 'N/A' }}
Prediction (s.d.)         {{ summary.average.predicted | round(3) }} ({{ summary.average.predicted_sd | round(2) }}){{ add_remaining_spaces(summary.average.predicted_length | default(0)) }}{{ summary.cumulative.predicted | round(3) if summary.cumulative else 'N/A' }} ({{ summary.cumulative.predicted_sd | round(2) if summary.cumulative else 'N/A' }})
{{ CI(alpha) }}                    [{{ summary.average.predicted_lower | round(3) }}, {{ summary.average.predicted_upper | round(3) }}]{{ add_remaining_spaces(4 + (summary.average.predicted_lower | round(3) | string | length) + (summary.average.predicted_upper | round(3) | string | length)) }}{{ "[{}, {}]".format(summary.cumulative.predicted_lower | round(3), summary.cumulative.predicted_upper | round(3)) if summary.cumulative else "N/A" }}

Absolute effect (s.d.)    {{ summary.average.abs_effect | round(3) }} ({{ summary.average.abs_effect_sd | round(2) }}){{ add_remaining_spaces(summary.average.abs_effect_length | default(0)) }}{{ summary.cumulative.abs_effect | round(3) if summary.cumulative else 'N/A' }} ({{ summary.cumulative.abs_effect_sd | round(2) if summary.cumulative else 'N/A' }})
{{ CI(alpha) }}                    [{{ summary.average.abs_effect_lower | round(3) }}, {{ summary.average.abs_effect_upper | round(3) }}]{{ add_remaining_spaces(4 + (summary.average.abs_effect_lower | round(3) | string | length) + (summary.average.abs_effect_upper | round(3) | string | length)) }}{{ "[{}, {}]".format(summary.cumulative.abs_effect_lower | round(3), summary.cumulative.abs_effect_upper | round(3)) if summary.cumulative else "N/A" }}

Relative effect (s.d.)    {{ '{0:.1%}'.format(summary.average.rel_effect) }} ({{ '{0:.1%}'.format(summary.average.rel_effect_sd | float) }}){{ add_remaining_spaces(3 + ('{0:.1%}'.format(summary.average.rel_effect) | length) + ('{0:.1%}'.format(summary.average.rel_effect_sd | float) | string | length)) }}{{ '{0:.1%}'.format(summary.cumulative.rel_effect) }} ({{ '{0:.1%}'.format(summary.cumulative.rel_effect_sd | round(2) | float) }}){{ "[{}, {}]".format(summary.cumulative.rel_effect_lower | round(3), summary.cumulative.rel_effect_upper | round(3)) if summary.cumulative else "N/A" }}

Posterior tail-area probability p: {{ p_value | round(3) }}
Posterior probability of an effect: {{ '{0:.2%}'.format(1 - p_value) }}

For more details run the command: summary(impact, output_format="report")
"""

report_text = """
{% macro CI(alpha) %}{{ (((1 - alpha) * 100) | string).rstrip('0').rstrip('.') }}%{% endmacro -%}
Analysis report {CausalImpact}

The model was run on data from 
{{ start_item }} 
to 
{{ end_item }}.
The post-intervention period started on 
{{ post_period_item }}.
{{ training_days_info }}

During the post-intervention period, the response variable had
an average value of approx. {{ summary.average.actual | round(3) }}. {% if detected_sig -%}By contrast, in{% else %}In{% endif %} the absence of an
intervention, we would have expected an average response of {{ summary.average.predicted | round(3) }}.
The {{ CI(alpha) }} interval of this counterfactual prediction is [{{ summary.average.predicted_lower | round(3) }}, {{ summary.average.predicted_upper | round(3) }}].
Subtracting this prediction from the observed response yields
an estimate of the causal effect the intervention had on the
response variable. This effect is {{ summary.average.abs_effect | round(3) }} with a {{ CI(alpha) }} interval of
[{{ summary.average.abs_effect_lower | round(3) }}, {{ summary.average.abs_effect_upper | round(3) }}]. For a discussion of the significance of this effect,
see below.

{% if summary.cumulative %}
Summing up the individual data points during the post-intervention
period (which can only sometimes be meaningfully interpreted), the
response variable had an overall value of {{ summary.cumulative.actual | round(3) }}.
{% if detected_sig %}By contrast, had{% else %}Had{% endif %} the intervention not taken place, we would have expected
a sum of {{ summary.cumulative.predicted | round(3) }}. The {{ CI(alpha) }} interval of this prediction is [{{ summary.cumulative.predicted_lower | round(3) }}, {{ summary.cumulative.predicted_upper | round(3) }}].
The difference between the actual and predicted sums is {{ cumulative_difference }}.
{% endif %}

The above results are given in terms of absolute numbers. In relative
terms, the response variable showed {% if positive_sig %}an increase of +{% else %}a decrease of {% endif %}{{ '{0:.1%}'.format(summary.average.rel_effect) }}. The {{ CI(alpha) }}
interval of this percentage is [{{ '{0:.1%}'.format(summary.average.rel_effect_lower) }}, {{ '{0:.1%}'.format(summary.average.rel_effect_upper) }}].
{% if detected_sig and positive_sig %}

This means that the positive effect observed during the intervention
period is statistically significant and unlikely to be due to random
fluctuations. It should be noted, however, that the question of whether
this increase also bears substantive significance can only be answered
by comparing the absolute effect ({{ summary.average.abs_effect | round(3) }}) to the original goal
of the underlying intervention.
{% elif detected_sig and not positive_sig %}

This means that the negative effect observed during the intervention
period is statistically significant.
If the experimenter had expected a positive effect, it is recommended
to double-check whether anomalies in the control variables may have
caused an overly optimistic expectation of what should have happened
in the response variable in the absence of the intervention.
{% elif not detected_sig and positive_sig %}

This means that, although the intervention appears to have caused a
positive effect, this effect is not statistically significant when
considering the entire post-intervention period as a whole. Individual
days or shorter stretches within the intervention period may of course
still have had a significant effect, as indicated whenever the lower
limit of the impact time series (lower plot) was above zero.
{% elif not detected_sig and not positive_sig -%}

This means that, although it may look as though the intervention has
exerted a negative effect on the response variable when considering
the intervention period as a whole, this effect is not statistically
significant and so cannot be meaningfully interpreted.
{% endif %}
{%- if not detected_sig %}

The apparent effect could be the result of random fluctuations that
are unrelated to the intervention. This is often the case when the
intervention period is very long and includes much of the time when
the effect has already worn off. It can also be the case when the
intervention period is too short to distinguish the signal from the
noise. Finally, failing to find a significant effect can happen when
there are not enough control variables or when these variables do not
correlate well with the response variable during the learning period.
{% endif %}
{%- if p_value < alpha %}

The probability of obtaining this effect by chance is very small
(Bayesian one-sided tail-area probability p = {{ p_value | round(3) }}).
This means the effect is statistically significant. It can be
considered causal if the model assumptions are satisfied.
{%- else %}

The probability of obtaining this effect by chance is p = {{ '{0:.0%}'.format(p_value) }}.
This means the effect may be spurious and would generally not be
considered statistically significant.
{%- endif %}

For more details, including the model assumptions behind the method, see
https://google.github.io/CausalImpact/.
"""

# Initialize the Jinja environment
def create_jinja_environment() -> Environment:
    """Create and configure the Jinja environment."""
    env = Environment(autoescape=False)
    env.globals['max'] = max  # Add max function
    env.globals['min'] = min  # Add min function
    return env

# Initialize templates using the environment
env = create_jinja_environment()
SUMMARY_TMPL = env.from_string(summary_text)
REPORT_TMPL = env.from_string(report_text)

def format_value(item: Optional[object]) -> str:
    """Format the item based on its type."""
    if item is None:
        return "no date info"
    elif isinstance(item, (int, float)):
        return f"{item}"
    elif isinstance(item, str):
        return item
    elif isinstance(item, datetime):
        return item.strftime('%Y-%m-%dT%H:%M:%SZ')
    else:
        return "no date info"

def get_index_item(index: list, position: int) -> Optional[datetime]:
    """Retrieve an item from the index based on the position."""
    if index and len(index) > 0:
        try:
            if position >= 0:
                return index[position]
            else:
                return index[len(index) + position]
        except IndexError:
            return None
    return None

def add_remaining_spaces(n: int) -> str:
    """Add remaining spaces for alignment."""
    return ' ' * max(19 - n, 0)

def format_relative_effect(rel_effect: float, rel_effect_sd: float) -> str:
    """Format the relative effect string."""
    direction = "an increase of +" if rel_effect > 0 else "a decrease of "
    return f"{direction}{rel_effect:.1%} ({rel_effect_sd:.1%})"

def format_p_value(p_value: float) -> str:
    """Format the p-value as a percentage."""
    return f"{p_value:.0%}"

def summary(
        ci_model: CausalImpactAnalysis, output_format: str = "summary", alpha: Optional[float] = None
):
    """Get summary of impact results.

    Args:
        ci_model: CausalImpact instance, after calling `.train`.
        output_format: string directing whether to print a shorter summary
          ('summary') or a long-form description ('report').
        alpha: float for alpha level to use; must be in (0, 1).

    Raises:
        DeprecationWarning: In case `alpha` is explicitly set.
        KeyError: If essential keys are missing in the summary data.

    Returns:
        Text output of summary results.
    """
    # Infer alpha from the model
    inferred_alpha = ci_model.summary.alpha.mean()
    if alpha is not None and alpha != inferred_alpha:
        raise DeprecationWarning("Supplying an argument to `alpha` is deprecated, "
                                 "since it is inferred from `ci_model`. Set "
                                 f"`alpha=None` to use alpha={inferred_alpha:.2f}, "
                                 f"or retrain the model with alpha={alpha}.")
    alpha = inferred_alpha

    # Validate output_format
    if output_format not in ["summary", "report"]:
        raise ValueError("`output_format` must be either 'summary' or 'report'. "
                         f"Got {output_format}")

    # Validate alpha
    if alpha <= 0. or alpha >= 1.:
        raise ValueError("`alpha` must be in (0, 1). Got %s" % alpha)

    # Extract p_value
    try:
        p_value = float(ci_model.summary["p_value"].iloc[0])
    except KeyError:
        raise KeyError("'p_value' key is missing in ci_model.summary")
    except (ValueError, TypeError):
        raise ValueError("'p_value' must be a numeric value")

    p_value_percentage = f"{p_value:.0%}"
    posterior_probability = f"{(1 - p_value):.2%}"

    # Transpose summary and convert to dict
    summary_data = ci_model.summary.transpose().to_dict()

    # Safely retrieve 'average' and 'cumulative', providing empty dicts if they don't exist
    average_data = summary_data.get('average', {})
    cumulative_data = summary_data.get('cumulative', {})

    # Format 'average' data (keeping as numbers)
    formatted_average = {
        "actual": float(average_data.get('actual')) if average_data.get('actual') is not None else None,
        "predicted": float(average_data.get('predicted')) if average_data.get('predicted') is not None else None,
        "predicted_sd": float(average_data.get('predicted_sd')) if average_data.get('predicted_sd') is not None else None,
        "predicted_lower": float(average_data.get('predicted_lower')) if average_data.get('predicted_lower') is not None else None,
        "predicted_upper": float(average_data.get('predicted_upper')) if average_data.get('predicted_upper') is not None else None,
        "abs_effect": float(average_data.get('abs_effect')) if average_data.get('abs_effect') is not None else None,
        "abs_effect_sd": float(average_data.get('abs_effect_sd')) if average_data.get('abs_effect_sd') is not None else None,
        "abs_effect_lower": float(average_data.get('abs_effect_lower')) if average_data.get('abs_effect_lower') is not None else None,
        "abs_effect_upper": float(average_data.get('abs_effect_upper')) if average_data.get('abs_effect_upper') is not None else None,
        "rel_effect": float(average_data.get('rel_effect', 0.0)),
        "rel_effect_sd": float(average_data.get('rel_effect_sd', 0.0)),
        "rel_effect_lower": float(average_data.get('rel_effect_lower', 0.0)),
        "rel_effect_upper": float(average_data.get('rel_effect_upper', 0.0)),
        # Calculate lengths for alignment
        "actual_length": len(f"{average_data.get('actual', 0.0):.3f}") if average_data.get('actual') is not None else 1,
        "predicted_length": (
            len(f"{average_data.get('predicted', 0.0):.3f}") +
            3 +  # for " ("
            len(f"{average_data.get('predicted_sd', 0.0):.2f}") +
            1    # for ")"
        ) if (average_data.get('predicted') is not None and
              average_data.get('predicted_sd') is not None) else 0,
        "predicted_ci_length": (
            len(f"{average_data.get('predicted_lower', 0.0):.3f}") +
            len(f"{average_data.get('predicted_upper', 0.0):.3f}") + 4  # for "[, ]"
        ) if (average_data.get('predicted_lower') is not None and
              average_data.get('predicted_upper') is not None) else 0,
        "abs_effect_length": (
            len(f"{average_data.get('abs_effect', 0.0):.3f}") +
            len(f"{average_data.get('abs_effect_sd', 0.0):.2f}") + 3  # for " ()"
        ) if (average_data.get('abs_effect') is not None and
              average_data.get('abs_effect_sd') is not None) else 0,
        "abs_effect_ci_length": (
            len(f"{average_data.get('abs_effect_lower', 0.0):.3f}") +
            len(f"{average_data.get('abs_effect_upper', 0.0):.3f}") + 4  # for "[, ]"
        ) if (average_data.get('abs_effect_lower') is not None and
              average_data.get('abs_effect_upper') is not None) else 0,
        "rel_effect_length": (
            len(f"{average_data.get('rel_effect', 0.0):.1%}") +
            len(f"{average_data.get('rel_effect_sd', 0.0):.1%}") + 3  # for " ()"
        ) if (average_data.get('rel_effect') is not None and
              average_data.get('rel_effect_sd') is not None) else 0,
        "rel_effect_ci_length": (
            len(f"{average_data.get('rel_effect_lower', 0.0):.1%}") +
            len(f"{average_data.get('rel_effect_upper', 0.0):.1%}") + 4  # for "[, ]"
        ) if (average_data.get('rel_effect_lower') is not None and
              average_data.get('rel_effect_upper') is not None) else 0,
    }

    # Similarly, ensure 'abs_effect_length' is set in 'formatted_cumulative'
    if cumulative_data:
        formatted_cumulative = {
            "actual": float(cumulative_data.get('actual')) if cumulative_data.get('actual') is not None else None,
            "predicted": float(cumulative_data.get('predicted')) if cumulative_data.get('predicted') is not None else None,
            "predicted_sd": float(cumulative_data.get('predicted_sd')) if cumulative_data.get('predicted_sd') is not None else None,
            "predicted_lower": float(cumulative_data.get('predicted_lower')) if cumulative_data.get('predicted_lower') is not None else None,
            "predicted_upper": float(cumulative_data.get('predicted_upper')) if cumulative_data.get('predicted_upper') is not None else None,
            "abs_effect": float(cumulative_data.get('abs_effect')) if cumulative_data.get('abs_effect') is not None else None,
            "abs_effect_sd": float(cumulative_data.get('abs_effect_sd')) if cumulative_data.get('abs_effect_sd') is not None else None,
            "abs_effect_lower": float(cumulative_data.get('abs_effect_lower')) if cumulative_data.get('abs_effect_lower') is not None else None,
            "abs_effect_upper": float(cumulative_data.get('abs_effect_upper')) if cumulative_data.get('abs_effect_upper') is not None else None,
            "rel_effect": float(cumulative_data.get('rel_effect', 0.0)),
            "rel_effect_sd": float(cumulative_data.get('rel_effect_sd', 0.0)),
            "rel_effect_lower": float(cumulative_data.get('rel_effect_lower', 0.0)),
            "rel_effect_upper": float(cumulative_data.get('rel_effect_upper', 0.0)),
            # Calculate lengths for alignment
            "actual_length": len(f"{cumulative_data.get('actual', 0.0):.3f}") if cumulative_data.get('actual') is not None else 1,
            "predicted_length": (
                len(f"{cumulative_data.get('predicted', 0.0):.3f}") +
                3 +  # for " ("
                len(f"{cumulative_data.get('predicted_sd', 0.0):.2f}") +
                1    # for ")"
            ) if (cumulative_data.get('predicted') is not None and
                  cumulative_data.get('predicted_sd') is not None) else 0,
            "predicted_ci_length": (
                len(f"{cumulative_data.get('predicted_lower', 0.0):.3f}") +
                len(f"{cumulative_data.get('predicted_upper', 0.0):.3f}") + 4  # for "[, ]"
            ) if (cumulative_data.get('predicted_lower') is not None and
                  cumulative_data.get('predicted_upper') is not None) else 0,
            "abs_effect_length": (
                len(f"{cumulative_data.get('abs_effect', 0.0):.3f}") +
                len(f"{cumulative_data.get('abs_effect_sd', 0.0):.2f}") + 3  # for " ()"
            ) if (cumulative_data.get('abs_effect') is not None and
                  cumulative_data.get('abs_effect_sd') is not None) else 0,
            "abs_effect_ci_length": (
                len(f"{cumulative_data.get('abs_effect_lower', 0.0):.3f}") +
                len(f"{cumulative_data.get('abs_effect_upper', 0.0):.3f}") + 4  # for "[, ]"
            ) if (cumulative_data.get('abs_effect_lower') is not None and
                  cumulative_data.get('abs_effect_upper') is not None) else 0,
            "rel_effect_length": (
                len(f"{cumulative_data.get('rel_effect', 0.0):.1%}") +
                len(f"{cumulative_data.get('rel_effect_sd', 0.0):.1%}") + 3  # for " ()"
            ) if (cumulative_data.get('rel_effect') is not None and
                  cumulative_data.get('rel_effect_sd') is not None) else 0,
            "rel_effect_ci_length": (
                len(f"{cumulative_data.get('rel_effect_lower', 0.0):.1%}") +
                len(f"{cumulative_data.get('rel_effect_upper', 0.0):.1%}") + 4  # for "[, ]"
            ) if (cumulative_data.get('rel_effect_lower') is not None and
                  cumulative_data.get('rel_effect_upper') is not None) else 0,
        }
    else:
        formatted_cumulative = {
            # Initialize all expected keys with default values
            "actual": None,
            "predicted": None,
            "predicted_sd": None,
            "predicted_lower": None,
            "predicted_upper": None,
            "abs_effect": None,
            "abs_effect_sd": None,
            "abs_effect_lower": None,
            "abs_effect_upper": None,
            "rel_effect": 0.0,
            "rel_effect_sd": 0.0,
            "rel_effect_lower": 0.0,
            "rel_effect_upper": 0.0,
            "actual_length": 1,
            "predicted_length": 0,
            "predicted_ci_length": 0,
            "abs_effect_length": 0,
            "abs_effect_ci_length": 0,
            "rel_effect_length": 0,
            "rel_effect_ci_length": 0,
        }

    formatted_summary = {
        "average": formatted_average,
        "cumulative": formatted_cumulative
    }

    # Prepare summary context
    summary_context = {
        "summary": formatted_summary,
        "alpha": alpha,
        "p_value": p_value,
        "posterior_probability": posterior_probability,
        "p_value_percentage": p_value_percentage,
        "add_remaining_spaces": add_remaining_spaces
    }

    # Render summary or report
    if output_format == "summary":
        output = SUMMARY_TMPL.render(
            summary=formatted_summary,
            alpha=alpha,
            p_value=p_value,
            posterior_probability=posterior_probability,
            p_value_percentage=p_value_percentage,
            add_remaining_spaces=add_remaining_spaces
        )
    else:
        # Preprocess data for report
        series = ci_model.series
        index = series.index.tolist() if hasattr(series, 'index') else []
        post_period_start = series.post_period_start.tolist() if hasattr(series, 'post_period_start') else []
        start_item_raw = get_index_item(index, 0)
        end_item_raw = get_index_item(index, -1)
        post_period_item_raw = post_period_start[0] if post_period_start else None

        # Format dates
        start_item = format_value(start_item_raw)
        end_item = format_value(end_item_raw)
        post_period_item = format_value(post_period_item_raw)

        # Calculate training days
        if isinstance(start_item_raw, datetime) and isinstance(post_period_item_raw, datetime):
            training_days = (post_period_item_raw - start_item_raw).days
            training_days_info = f"A total of {training_days} days were used for training the model."
        else:
            training_days_info = "A total of no training days information available."

        # Determine significance and effect direction
        rel_effect_lower = summary_data.get('average', {}).get('rel_effect_lower', 0.0)
        rel_effect_upper = summary_data.get('average', {}).get('rel_effect_upper', 0.0)
        rel_effect = summary_data.get('average', {}).get('rel_effect', 0.0)

        detected_sig = not (rel_effect_lower < 0 and rel_effect_upper > 0)
        positive_sig = rel_effect > 0

        # Calculate cumulative difference
        if cumulative_data and ('actual' in cumulative_data and 'predicted' in cumulative_data):
            try:
                cumulative_actual = float(cumulative_data.get('actual', 0.0))
                cumulative_predicted = float(cumulative_data.get('predicted', 0.0))
                cumulative_difference = cumulative_actual - cumulative_predicted
                cumulative_difference = f"{cumulative_difference:.3f}"
            except (TypeError, ValueError):
                cumulative_difference = "N/A"
        else:
            cumulative_difference = "N/A"

        # Format relative effect
        average_rel_effect = summary_data.get('average', {}).get('rel_effect', 0.0)
        average_rel_effect_sd = summary_data.get('average', {}).get('rel_effect_sd', 0.0)
        relative_effect = format_relative_effect(average_rel_effect, average_rel_effect_sd)

        # Prepare the context for the report template
        report_context = {
            "start_item": start_item,
            "end_item": end_item,
            "post_period_item": post_period_item,
            "training_days_info": training_days_info,
            "summary": formatted_summary,
            "alpha": alpha,
            "p_value": p_value,
            "p_value_percentage": p_value_percentage,
            "posterior_probability": posterior_probability,
            "detected_sig": detected_sig,
            "positive_sig": positive_sig,
            "cumulative_difference": cumulative_difference,
            "relative_effect": relative_effect,
            # 'CI' macro is handled within the template; no need to pass it here
        }

        # Render report template
        output = REPORT_TMPL.render(report_context)

    # Log the formatted summary
    logger.debug(f"Formatted Summary: {formatted_summary}")

    return output


