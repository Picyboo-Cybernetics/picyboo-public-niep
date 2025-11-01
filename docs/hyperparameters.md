# NIEP Hyperparameters

The following tables summarise the default ranges recommended in the
"Neuro-Inspired Error Propagation (NIEP)" whitepaper.  They provide practical
starting points for practitioners experimenting with continual-learning
benchmarks.  The defaults implemented in `src/python/niep_reference.py`
correspond to the *balanced* profile.

## Core dynamics

| Symbol | Name                 | Description                                                                 | Typical range | Default |
|--------|----------------------|-----------------------------------------------------------------------------|---------------|---------|
| $\alpha$ | Refractory decay     | Controls how quickly a saturated refractory trace relaxes.                 | 0.7–0.95      | 0.9     |
| $\beta$  | Activation coupling  | Contribution of the absolute activation to the refractory increment.       | 0.01–0.15     | 0.05    |
| $\gamma$ | Gradient coupling    | Contribution of the gradient magnitude to the refractory increment.       | 0.01–0.15     | 0.05    |
| $\kappa$ | Gate threshold       | Refractory level where the logistic gate is half-open.                     | 0.2–0.4       | 0.3     |
| $T$       | Gate temperature     | Smoothness of the refractory gate.                                         | 0.01–0.1      | 0.05    |
| $\lambda$ | Eligibility decay    | Weight assigned to the previous eligibility trace.                         | 0.7–0.99      | 0.9     |
| $\rho$    | Budget decay         | Exponential smoothing factor for the error budget.                         | 0.6–0.95      | 0.9     |
| $\delta$  | Budget scale         | Translating eligibility magnitude into additional budget.                  | 0.1–2.0       | 1.0     |
| $\xi$     | Budget floor         | Minimum update capacity to avoid dead weights.                             | 0.01–0.2      | 0.1     |
| $\chi$    | Commit blend         | Portion of the shadow weights merged during a successful commit.          | 0.1–0.5       | 0.25    |

## Scenario presets

### Continual supervised learning

* Higher $\lambda$ (0.9–0.99) to retain longer gradient history.
* Conservative $\chi$ (0.1–0.2) to minimise catastrophic interference.
* Monitor calibration metrics (ECE, NLL) alongside accuracy.

### Federated / asynchronous training

* Tighten validation windows (every 5–10 local steps).
* Increase $\xi$ to at least 0.15 to maintain progress despite sparse updates.
* Track per-client budgets and aggregate them using the harmonic mean to avoid
  dominance by a single node.

### Resource constrained edge devices

* Lower $\lambda$ (0.7–0.85) and $\rho$ (0.6–0.8) to reduce memory footprint.
* Periodically zero-out refractory traces for inactive channels to recycle
  capacity.
* Schedule safe-commit validations less frequently but use multiple metrics
  (e.g. loss + drift score) before accepting updates.

## Monitoring checklist

* Visualise the refractory histogram to ensure the majority of parameters spend
  <50% of the time in a saturated state.
* Track the ratio of accepted to rejected commits.  Persistent rejections often
  signal that budgets or gate thresholds require adjustment.
* Record the effective learning rate `budget * gate` to catch units that may be
  starved due to interacting hyperparameters.
