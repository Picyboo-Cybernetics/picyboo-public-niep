# NIEP: Neuro-Inspired Error Propagation

**Series:** Picyboo Public Research Series  
**Organization:** Picyboo Cybernetics Inc., Research Lab (Canada)

**Keywords:** Continual Learning, Refractory Gating, Eligibility Traces, Catastrophic Interference, Federated Learning, Asynchronous Training, Energy-Efficient AI, Stability-Plasticity Dilemma

## About Picyboo Cybernetics

We develop advanced systems across quantum computing, artificial intelligence, and decentralized networks to enable the next generation of technology. Therefore, we distribute select frameworks, implementations, and development tools as open source—enabling developers and institutions to build on our technology foundation.

## Overview

NIEP augments backpropagation with neuro-inspired mechanisms that stabilise continual learning. The repository now ships a framework-agnostic reference implementation, reproducible examples and operational guidance that complement the whitepaper.

## Whitepaper

- PDF: docs/halenta-neuro-inspired-error-propagation-(NIEP)-2025-10-11-v1.3.pdf  
- DOI: https://doi.org/10.5281/zenodo.17455451

## Repository Purpose

Public research reference for industry and academic collaborators. Provides production-ready reference implementations and operational guidance for deploying NIEP-based continual learning systems.

## Repository Highlights

- `src/python/niep_reference.py`: numerically stable implementations of the refractory, gating, eligibility and safe-commit primitives.
- `examples/`: runnable NumPy notebook-style scripts illustrating how to plug the NIEP components into a continual learning loop.
- `docs/hyperparameters.md`: practitioner-focused parameter ranges and presets for diverse deployment scenarios.
- `docs/safety.md`: operational safeguards and monitoring guidance for the safe-commit protocol.
- `ROADMAP.md`: forward-looking milestones covering research, tooling and theoretical analysis.

## Installation & Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pytest
```

> The reference module is framework agnostic. Replace `numpy` with `torch` or `jax` depending on your stack and adapt the tensor handling accordingly.

## Quickstart

1. Inspect and modify `examples/simple_niep_loop.py` to experiment with the gating and budget dynamics on synthetic data.
2. Integrate `refractory_update`, `gating`, `eligibility_update` and `budget_update` into your training loop using your framework of choice.
3. Use `safe_commit` to blend shadow weights once validation diagnostics indicate progress. The helper expects main/shadow metric dictionaries and returns both the updated parameters and a commit flag.

## Testing

Run the unit tests to validate numerical behaviour:

```bash
pytest
```

The tests cover gating monotonicity, eligibility accumulation, budget floors and safe-commit decision logic.

## Status

Openly published for transparency. Framework-agnostic implementation available with comprehensive testing and documentation.

## Contributing

Community feedback and pull requests are welcome. Please review the roadmap and open an issue describing the feature or evaluation benchmark you would like to tackle.

## License

This repository is released under the Apache 2.0 License; see `LICENSE` for full terms.

## How to Cite

> Halenta, D. N. (2025). *Neuro-Inspired Error Propagation (NIEP): Refractory Period-Based Learning for Stable, Incremental AI Systems.*  
> Picyboo Cybernetics Inc.  
> DOI: https://doi.org/10.5281/zenodo.17455451

## Links

- Website: https://picyboo.com
- Technical Sandbox: https://picyboo.net
- GitHub Organization: https://github.com/Picyboo-Cybernetics