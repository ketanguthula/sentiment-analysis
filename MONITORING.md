# Drift Monitoring Analysis

## Overview

This project includes an Evidently-based drift monitoring script located at:

```text
src/monitor_drift.py 

## Monitoring Frequency

The drift monitoring script should be executed regularly in production. A reasonable starting point would be daily or weekly monitoring, depending on application traffic volume.

## Script Output

The drift script is designed to exit with a nonzero status when drift exceeds the configured threshold.

## Retraining Strategy

If substantial drift is detected, the following process is recommended:

1. Investigate the drifted features.
2. Review recent prediction examples.
3. Collect representative production data.
4. Re-label newly collected samples if necessary.
5. Retrain the sentiment classifier.
6. Compare the updated model against the current production model using MLflow experiments.
7. Deploy the improved model if performance increases.

This process would help maintain model accuracy as language patterns evolve over time. 