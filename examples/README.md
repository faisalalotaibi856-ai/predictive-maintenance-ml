# Prediction example

`machine_sample.json` contains exactly the six input features from the original UCI CSV row `UDI=1`. Identifiers, target and failure-mode flags are excluded. It is an interface example, not a new independent evaluation observation.

From the repository root:

```bash
python -m src.predict --sample-file examples/machine_sample.json
```

For another sample, create a JSON object with the same six keys. Use Type L/M/H and the units in each key (temperatures are kelvin, not Celsius). Numeric fields must be valid finite values. The output's `failure_probability` is an uncalibrated model score; `predicted_class` applies the saved validation-selected threshold.
