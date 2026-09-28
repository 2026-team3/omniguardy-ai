# Door Event model contract

## Classes and events

| Class | Index | Spring event |
| --- | ---: | --- |
| background | 0 | NO_EVENT |
| knock | 1 | KNOCK_EVENT |
| handle | 2 | HANDLE_EVENT |

The model output is a three-value softmax vector. Knock and Handle thresholds are
calibrated independently on validation data. Test data is opened only after the model
and both thresholds are fixed.

## Real-world evaluation

Continuous recordings require reference event start/end timestamps. A detection is
matched at most once to a reference event of the same class. Report event precision,
event recall, false triggers per hour, mean latency and p95 latency. Cooldown is an
operational duplicate-suppression parameter and must not be used to hide false
positives during threshold calibration.
