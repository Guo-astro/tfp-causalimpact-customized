# tfp_causalimpact_customized

## Features

- Rebuilt of [TFP CausalImpact](https://github.com/google/tfp-causalimpact)

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
