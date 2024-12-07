# tfp_causalimpact_customized

## Features


Highlighting Missing Post-Intervention Observations:

In scenarios where no observed data exist for certain time points within the post-intervention period, the forecasted values are still computed by the model but cannot be validated against observed outcomes. To visually distinguish these points in the plots, we highlight them differently (e.g., using a dashed line and a separate color). This approach ensures that readers can easily identify which portions of the forecast are based purely on model inference (no ground-truth observations available) and which are directly comparable to actual observed data. This visual cue can be critical for interpreting the reliability and meaning of the estimated causal effects during periods with missing observations.

- Rebuilt of [TFP CausalImpact](https://github.com/google/tfp-causalimpact)

### Improved summary round to 3 digits

### Matplotlib Japanese Support

- Added support for Japanese fonts and characters in Matplotlib plots.
- Enhanced compatibility with Japanese data visualization requirements.

### Improved Matplotlib Plots

- Enhanced plotting capabilities for clearer and more informative visualizations.
- Customized plot styles and themes to better represent causal impact analysis.

## Comparison with [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact)

### Enhancements Over tfcausalimpact

- **Stability:** Resolved the issue of results changing from run to run, ensuring consistent outcomes.
  See [Result change from run to run in tfcausalimpact](https://stackoverflow.com/questions/69257795/result-change-from-run-to-run-in-tfcausalimpact).
- **Performance:** Optimized performance for faster computations and larger datasets.
- **Customization:** Increased flexibility in model customization and parameter tuning.

### Fixed Issues

- **Consistent Results:** Fixed
  the [Result change from run to run in tfcausalimpact](https://stackoverflow.com/questions/69257795/result-change-from-run-to-run-in-tfcausalimpact)
  issue to ensure reproducible results across multiple runs.
- **Bug Fixes:** Addressed various bugs reported in the
  original [tfcausalimpact](https://github.com/WillianFuks/tfcausalimpact) repository to enhance overall stability and
  reliability.

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