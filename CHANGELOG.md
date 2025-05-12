# 3.0.1 (2025 - 05 - 12)

### Features

* Support for python 3.12 and 3.13. Drop support 3.7 and 3.8.
* Add `history_` attribute to track loss during training
* Add support for [`BaseConfig`](https://tno-quantum.github.io/documentation/content/packages/packages/tno.quantum.utils/main.html#tno.quantum.utils.BaseConfig) arguments from [`tno.quantum.utils`](https://tno-quantum.github.io/documentation/content/packages/packages/tno.quantum.utils/main.html)

### Bug fixes

* Torch concatenate errors for python3.12

### BREAKING CHANGES

* **Rename models:** `ProbabilitiesModel` to `ProbabilityModel` and `ExpectedValuesModel` to `ExpectedValueModel`.


# 2.0.2 (2023 - 07 - 06)

### Features

* **Parity Model:** Add new post-processing model. 


### BREAKING CHANGES

* **Post-processing model:** Renamed `probabilities_model` to `modulo_model`.


# 1.2.0 (2023 - 02 - 13)

* **Initial public release:** Scikit-learn compliant Variational Quantum Classifier.