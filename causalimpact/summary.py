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

"""Utils for printing impact summaries."""

from typing import Optional
from jinja2 import Environment, Template
from datetime import datetime
from causalimpact.causalimpact_lib import CausalImpactAnalysis

# Define the summary template
summary_text = """
{% macro CI(alpha) %}{{(((1 - alpha) * 100) | string).rstrip('0').rstrip('.')}}% CI{% endmacro -%}
{% macro add_remaining_spaces(n) %}{{' ' * (19 -n)}}{% endmacro -%}
Posterior Inference {CausalImpact}
                          Average            Cumulative
Actual                    {{summary.average.actual}} {{add_remaining_spaces(summary.average.actual_length)}}{{summary.cumulative.actual}}
Prediction (s.d.)         {{summary.average.predicted}} ({{summary.average.predicted_sd}}){{add_remaining_spaces(summary.average.predicted_length)}}{{summary.cumulative.predicted}} ({{summary.cumulative.predicted_sd}})
{{CI(alpha)}}                    [{{summary.average.predicted_lower}}, {{summary.average.predicted_upper}}]{{add_remaining_spaces(summary.predicted_ci_length)}}[{{summary.cumulative.predicted_lower}}, {{summary.cumulative.predicted_upper}}]

Absolute effect (s.d.)    {{summary.average.abs_effect}} ({{summary.average.abs_effect_sd}}){{add_remaining_spaces(summary.abs_effect_length)}}{{summary.cumulative.abs_effect}} ({{summary.cumulative.abs_effect_sd}})
{{CI(alpha)}}                    [{{summary.average.abs_effect_lower}}, {{summary.average.abs_effect_upper}}]{{add_remaining_spaces(summary.abs_effect_ci_length)}}[{{summary.cumulative.abs_effect_lower}}, {{summary.cumulative.abs_effect_upper}}]

Relative effect (s.d.)    {{summary.average.rel_effect}} ({{summary.average.rel_effect_sd}}){{add_remaining_spaces(summary.rel_effect_length)}}{{summary.cumulative.rel_effect}} ({{summary.cumulative.rel_effect_sd}})
{{CI(alpha)}}                    [{{summary.average.rel_effect_lower}}, {{summary.average.rel_effect_upper}}]{{add_remaining_spaces(summary.rel_effect_ci_length)}}[{{summary.cumulative.rel_effect_lower}}, {{summary.cumulative.rel_effect_upper}}]

Posterior tail-area probability p: {{p_value}}
Posterior probability of an effect: {{posterior_probability}}

For more details run the command: summary(impact, output_format="report")
"""

# Define the report template
report_text = """
The model was run on data from {{start_item}} to {{end_item}}.
The post-intervention period started on {{post_period_item}}.
{{training_days_info}}

Analysis report {CausalImpact}

During the post-intervention period, the response variable had
an average value of approx. {{summary.average.actual}}. {{contrast_text}} the absence of an
intervention, we would have expected an average response of {{summary.average.predicted}}.
The {{CI(alpha)}} interval of this counterfactual prediction is [{{summary.average.predicted_lower}}, {{summary.average.predicted_upper}}].
Subtracting this prediction from the observed response yields
an estimate of the causal effect the intervention had on the
response variable. This effect is {{summary.average.abs_effect}} with a {{CI(alpha)}} interval of
[{{summary.average.abs_effect_lower}}, {{summary.average.abs_effect_upper}}]. For a discussion of the significance of this effect,
see below.

Summing up the individual data points during the post-intervention
period (which can only sometimes be meaningfully interpreted), the
response variable had an overall value of {{summary.cumulative.actual}}.
{{intervention_contrast_text}} the intervention not taken place, we would have expected
a sum of {{summary.cumulative.predicted}}. The {{CI(alpha)}} interval of this prediction is [{{summary.cumulative.predicted_lower}}, {{summary.cumulative.predicted_upper}}].
The difference between the actual and predicted sums is {{cumulative_difference}}.

The above results are given in terms of absolute numbers. In relative
terms, the response variable showed {{relative_effect}}. The {{CI(alpha)}}
interval of this percentage is [{{summary.average.rel_effect_lower}}, {{summary.average.rel_effect_upper}}].
{% if detected_sig and positive_sig %}

This means that the positive effect observed during the intervention
period is statistically significant and unlikely to be due to random
fluctuations. It should be noted, however, that the question of whether
this increase also bears substantive significance can only be answered
by comparing the absolute effect ({{summary.average.abs_effect}}) to the original goal
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
{% elif not detected_sig and not positive_sig %}
This means that, although it may look as though the intervention has
exerted a negative effect on the response variable when considering
the intervention period as a whole, this effect is not statistically
significant and so cannot be meaningfully interpreted.
{% endif %}
{% if not detected_sig %}

The apparent effect could be the result of random fluctuations that
are unrelated to the intervention. This is often the case when the
intervention period is very long and includes much of the time when
the effect has already worn off. It can also be the case when the
intervention period is too short to distinguish the signal from the
noise. Finally, failing to find a significant effect can happen when
there are not enough control variables or when these variables do not
correlate well with the response variable during the learning period.
{% endif %}
{% if p_value < alpha %}

The probability of obtaining this effect by chance is very small
(Bayesian one-sided tail-area probability p = {{p_value}}).
This means the effect is statistically significant. It can be
considered causal if the model assumptions are satisfied.
{% else %}

The probability of obtaining this effect by chance is p = {{p_value_percentage}}.
This means the effect may be spurious and would generally not be
considered statistically significant.
{% endif %}

For more details, including the model assumptions behind the method, see
https://google.github.io/CausalImpact/.
"""

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
    return ' ' * (19 - n)

def format_relative_effect(rel_effect: float, rel_effect_sd: float) -> str:
    """Format the relative effect string."""
    direction = "an increase of +" if rel_effect > 0 else "a decrease of "
    return f"{direction}{rel_effect:.1%} ({rel_effect_sd:.1%})"

def format_p_value(p_value: float) -> str:
    """Format the p-value as a percentage."""
    return f"{p_value:.0%}"

def create_jinja_environment() -> Environment:
    """Create and configure the Jinja environment."""
    env = Environment(autoescape=False)
    return env

# Initialize the Jinja environment
env = create_jinja_environment()
SUMMARY_TMPL = env.from_string(summary_text)
REPORT_TMPL = env.from_string(report_text)

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

    Returns:
        Text output of summary results.
    """
    inferred_alpha = ci_model.summary.alpha.mean()
    if alpha is not None and alpha != inferred_alpha:
        raise DeprecationWarning("Supplying an argument to `alpha` is deprecated, "
                                 "since it is inferred from `ci_model`. Set "
                                 f"`alpha=None` to use alpha={inferred_alpha:.2f}, "
                                 f"or retrain the model with alpha={alpha}.")
    alpha = inferred_alpha

    if output_format not in ["summary", "report"]:
        raise ValueError("`output_format` must be either 'summary' or 'report'. "
                         f"Got {output_format}")

    if alpha <= 0. or alpha >= 1.:
        raise ValueError("`alpha` must be in (0, 1). Got %s" % alpha)

    p_value = ci_model.summary["p_value"].iloc[0]
    p_value_percentage = f"{p_value:.0%}"
    posterior_probability = f"{(1 - p_value):.2%}"

    # Format summary data
    summary_data = ci_model.summary.transpose().to_dict()

    # Preprocess summary data for the summary template
    formatted_summary = {
        "average": {
            "actual": f"{summary_data['average']['actual']:.3f}",
            "predicted": f"{summary_data['average']['predicted']:.3f}",
            "predicted_sd": f"{summary_data['average']['predicted_sd']:.2f}",
            "predicted_lower": f"{summary_data['average']['predicted_lower']:.3f}",
            "predicted_upper": f"{summary_data['average']['predicted_upper']:.3f}",
            "abs_effect": f"{summary_data['average']['abs_effect']:.3f}",
            "abs_effect_sd": f"{summary_data['average']['abs_effect_sd']:.2f}",
            "abs_effect_lower": f"{summary_data['average']['abs_effect_lower']:.3f}",
            "abs_effect_upper": f"{summary_data['average']['abs_effect_upper']:.3f}",
            "rel_effect": f"{summary_data['average']['rel_effect']:.1%}",
            "rel_effect_sd": f"{summary_data['average']['rel_effect_sd']:.1%}",
            "rel_effect_lower": f"{summary_data['average']['rel_effect_lower']:.1%}",
            "rel_effect_upper": f"{summary_data['average']['rel_effect_upper']:.1%}",
        },
        "cumulative": {
            "actual": f"{summary_data['cumulative']['actual']:.3f}",
            "predicted": f"{summary_data['cumulative']['predicted']:.3f}",
            "predicted_sd": f"{summary_data['cumulative']['predicted_sd']:.2f}",
            "predicted_lower": f"{summary_data['cumulative']['predicted_lower']:.3f}",
            "predicted_upper": f"{summary_data['cumulative']['predicted_upper']:.3f}",
            "abs_effect": f"{summary_data['cumulative']['abs_effect']:.3f}",
            "abs_effect_sd": f"{summary_data['cumulative']['abs_effect_sd']:.2f}",
            "abs_effect_lower": f"{summary_data['cumulative']['abs_effect_lower']:.3f}",
            "abs_effect_upper": f"{summary_data['cumulative']['abs_effect_upper']:.3f}",
            "rel_effect": f"{summary_data['cumulative']['rel_effect']:.1%}",
            "rel_effect_sd": f"{summary_data['cumulative']['rel_effect_sd']:.1%}",
            "rel_effect_lower": f"{summary_data['cumulative']['rel_effect_lower']:.1%}",
            "rel_effect_upper": f"{summary_data['cumulative']['rel_effect_upper']:.1%}",
        }
    }

    # Precompute lengths for alignment in summary template
    # (Assuming fixed spacing, these could be adjusted as needed)
    formatted_summary['average']['actual_length'] = len(formatted_summary['average']['actual'])
    formatted_summary['average']['predicted_length'] = len(formatted_summary['average']['predicted']) + 3 + len(formatted_summary['average']['predicted_sd'])
    formatted_summary['predicted_ci_length'] = len(formatted_summary['average']['predicted_lower']) + len(formatted_summary['average']['predicted_upper']) + 4
    formatted_summary['abs_effect_length'] = len(formatted_summary['average']['abs_effect']) + len(formatted_summary['average']['abs_effect_sd']) + 3
    formatted_summary['abs_effect_ci_length'] = len(formatted_summary['average']['abs_effect_lower']) + len(formatted_summary['average']['abs_effect_upper']) + 4
    formatted_summary['rel_effect_length'] = len(formatted_summary['average']['rel_effect']) + len(formatted_summary['average']['rel_effect_sd']) + 3
    formatted_summary['rel_effect_ci_length'] = len(formatted_summary['average']['rel_effect_lower']) + len(formatted_summary['average']['rel_effect_upper']) + 4

    # Render summary template
    if output_format == "summary":
        output = SUMMARY_TMPL.render(
            summary=formatted_summary,
            alpha=alpha,
            p_value=p_value,
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
        if start_item_raw and post_period_item_raw:
            training_days = (post_period_item_raw - start_item_raw).days
            training_days_info = f"A total of {training_days} days were used for training the model."
        else:
            training_days_info = "A total of no training days information available."

        # Determine significance and effect direction
        detected_sig = not (summary_data['average']['rel_effect_lower'] < 0 and summary_data['average']['rel_effect_upper'] > 0)
        positive_sig = summary_data['average']['rel_effect'] > 0

        # Prepare contrast texts
        contrast_text = "By contrast," if detected_sig else "In"
        intervention_contrast_text = "By contrast, had" if detected_sig else "Had"

        # Calculate cumulative difference
        cumulative_difference = f"{(ci_model.summary['cumulative']['actual'] - ci_model.summary['cumulative']['predicted']).iloc[0]:.3f}"

        # Format relative effect
        relative_effect = format_relative_effect(summary_data['average']['rel_effect'], summary_data['average']['rel_effect_sd'])

        # Prepare the context for the report template
        report_context = {
            "start_item": start_item,
            "end_item": end_item,
            "post_period_item": post_period_item,
            "training_days_info": training_days_info,
            "summary": formatted_summary,
            "alpha": alpha,
            "p_value": f"{p_value:.3f}",
            "p_value_percentage": p_value_percentage,
            "posterior_probability": posterior_probability,
            "detected_sig": detected_sig,
            "positive_sig": positive_sig,
            "contrast_text": contrast_text,
            "intervention_contrast_text": intervention_contrast_text,
            "cumulative_difference": cumulative_difference,
            "relative_effect": relative_effect,
            "CI": lambda a: f"{((1 - a) * 100):.0f}% CI"
        }

        # Render report template
        output = REPORT_TMPL.render(report_context)

    return output
