# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.1.29

### Added

- **Helper Functions:**
    - Introduced `process_y_formatter_units` to handle flexible y-axis formatter units.
    - Introduced `create_y_axis_formatter` to generate y-axis formatter functions based on user input.
    - Introduced `add_period_markers` to add vertical period markers to each subplot, reducing code duplication.

- **Documentation Enhancements:**
    - Expanded docstrings for the main function and all helper functions to provide detailed descriptions of their
      purposes, parameters, and return values.
    - Added comprehensive inline comments to demarcate different sections of the code and explain complex logic.

### Changed

- **Y-Axis Formatting:**
    - Enhanced y-axis formatting flexibility by allowing distinct formatter units for each subplot.
    - Implemented dynamic y-axis formatter creation based on user-specified options.

### Fixed

- **Error Handling:**
    - Added error checks in `process_y_formatter_units` to ensure:
        - The length of the `y_formatter_unit` list matches the number of `y_labels`.
        - The input type for `y_formatter_unit` is valid (string, list, or dictionary).
    - These checks prevent potential runtime errors and provide informative feedback to the user.

