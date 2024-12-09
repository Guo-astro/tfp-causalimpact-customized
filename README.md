# tfp_causalimpact_customized

#### A stable multi-chain rebuilt of [TFP CausalImpact](https://github.com/google/tfp-causalimpact)

## Features

- Added convergence tests to the Gibbs sampling process.

- Improved summary round to 3 digits
- Added support for Japanese fonts and characters in Matplotlib plots.
- Enhanced compatibility with Japanese data visualization requirements.
- Enhanced plotting capabilities for chain convergence visualizations.

- Enhancements Over tfcausalimpact [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact)
    - **Stability:** Resolved the issue of results changing from run to run, ensuring consistent outcomes.
      See [Result change from run to run in tfcausalimpact](https://stackoverflow.com/questions/69257795/result-change-from-run-to-run-in-tfcausalimpact).

## Getting Started

1. **Installation**
   ```bash
   uv add tfp_causalimpact_customized
   ```
2. **Plot options** (Currently only Matplotlib is supported)
   Important:y_formatter_unit must be a dictionary with the **keys** that are the same as legend_labels **y_labels**.

```python
plot_options = {
    'chart_width': 1000,
    'chart_height': 200,
    'x_label': 'Date',
    'y_labels': ['Observed1', 'Pointwise Effect1', 'Cumulative Effect1'],
    'title': 'Customized Matplotlib Plot',
    'title_font_size': 16,
    'axis_title_font_size': 14,
    'y_formatter': 'millions',
    'y_formatter_unit': {
        'Observed1': ' units',
        'Pointwise Effect1': ' effect',
        'Cumulative Effect1': ' total'
    },
    'legend_labels': {
        'mean': 'Average',
        'observed': 'Observed',
        'pointwise': 'Pointwise Effect',
        'cumulative': 'Cumulative Effect',
        'pre-period-start': 'Start of Pre-Period',
        'pre-period-end': 'End of Pre-Period',
        'post-period-start': 'Start of Post-Period',
        'post-period-end': 'End of Post-Period'
    }
}
   ```