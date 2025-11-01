# NIEP Roadmap

The roadmap translates the open research questions from the NIEP whitepaper
into actionable milestones.  It is split into short-, mid- and long-term goals
covering implementation, evaluation and theoretical analysis.

## Q2 2025 – Reference implementation & tooling

- [ ] Release PyTorch and JAX back-ends wrapping the reference functions in
      `src/python/niep_reference.py`.
- [ ] Publish training scripts for Split-MNIST, Permuted-CIFAR and Omniglot
      continual-learning benchmarks.
- [ ] Provide precomputed baselines comparing NIEP to SGD, Adam, EWC and SI.
- [ ] Package the metrics suite (SPI, average forgetting, calibration) as a
      reusable `niep-metrics` Python module.

## Q3 2025 – Robustness & deployment readiness

- [ ] Implement differential privacy aware variants of the error budget.
- [ ] Extend safe-commit with Byzantine-resilient aggregation for federated
      training.
- [ ] Integrate streaming drift detectors (ADWIN, Page-Hinkley) and define
      escalation policies when drift coincides with commit rejections.
- [ ] Release infrastructure-as-code templates for running NIEP experiments on
      managed GPU clusters.

## Q4 2025 – Theoretical analysis & neuromorphic exploration

- [ ] Formalise convergence guarantees under bounded gradient variance.
- [ ] Investigate compressed state representations to reduce the 4× memory
      overhead highlighted in the whitepaper.
- [ ] Prototype a neuromorphic implementation leveraging event-driven hardware.
- [ ] Launch a community benchmark challenge featuring incremental robotics
      control tasks.

## Continuous contributions

- [ ] Maintain the hyperparameter knowledge base (`docs/hyperparameters.md`).
- [ ] Curate safety incident reports and responses in `docs/safety.md`.
- [ ] Collect user feedback from the examples/ directory and expand coverage to
      reinforcement learning and sequence modelling use cases.
