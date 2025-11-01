# Safety & Reliability Guidelines

The safe-commit protocol is designed to prevent catastrophic regressions in
continual-learning deployments.  This document condenses the operational
recommendations from Sections 10–12 of the NIEP whitepaper.

## Validation windows

* Perform validation every 5–20 tentative updates depending on task volatility.
* Use disjoint validation buffers for different metrics (e.g. loss vs. drift) to
  reduce correlated failures.
* Maintain a rolling baseline of reference metrics so that noisy improvements do
  not trigger commits.

## Multi-metric commit criteria

1. Define a monitoring schema covering at least **loss**, **calibration**
   (expected calibration error) and **stability** (e.g. prediction variance).
2. Require shadow weights to improve all *lower-is-better* metrics and not
   degrade any *higher-is-better* metrics before accepting a commit.
3. In high-risk environments, add a `min_delta` threshold rather than accepting
   marginal gains.

## Audit logging

* Record each validation window, including hashes of the evaluated datasets,
  metric values and commit decisions.
* Keep the shadow parameters on disk for a configurable retention period to
  support post-mortem analysis.
* Emit structured events (JSON/Protobuf) to integrate with security information
  and event management (SIEM) systems.

## Adversarial considerations

* Randomise validation ordering and maintain a reserve validation set to limit
  the effectiveness of data-poisoning attacks.
* Sign validation payloads and confirm their provenance when operating across
  federated or untrusted networks.
* Monitor the ratio of rejected commits—sudden spikes may indicate adversarial
  attempts to exhaust budgets or induce drift.

## Failsafe modes

* Introduce a watchdog that halts safe-commit operations if the variance of the
  monitored metrics exceeds a configurable bound.
* Provide a manual override that can pin the production parameters to a known
  good checkpoint regardless of shadow updates.
* Combine NIEP with traditional drift detectors to trigger human review when the
  environment distribution shifts abruptly.
