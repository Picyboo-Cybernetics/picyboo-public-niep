"""Minimal NIEP training loop demonstration.

The script keeps the computations fully in NumPy for reproducibility and
serves purely as an educational reference.  It demonstrates one iteration of
refractory gating, eligibility trace accumulation, budget tracking and the safe
commit protocol using synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.python.niep_reference import (
    NIEPState,
    budget_update,
    eligibility_update,
    gating,
    refractory_update,
    safe_commit,
)


@dataclass
class ToyModel:
    params: Dict[str, np.ndarray]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.params["w"] + self.params["b"]

    def gradients(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        preds = self.forward(x)
        error = preds - y
        grad_w = x.T @ error / len(x)
        grad_b = np.mean(error, axis=0, keepdims=True)
        return {"w": grad_w, "b": grad_b}


def main(seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    # Synthetic linear regression task with drifting target weights.
    x = rng.normal(size=(32, 4))
    target_w = np.linspace(0.1, 1.0, 4)
    target_b = np.array([[0.05]])
    noise = rng.normal(scale=0.01, size=(32, 1))
    y = x @ target_w[:, None] + target_b + noise

    model = ToyModel(params={"w": rng.normal(scale=0.1, size=(4, 1)), "b": np.zeros((1, 1))})
    shadow = {k: v.copy() for k, v in model.params.items()}

    state = NIEPState(
        r=np.zeros_like(model.params["w"]),
        e=np.zeros_like(model.params["w"]),
        B=np.ones_like(model.params["w"]),
        w_tilde=shadow,
    )

    grads = model.gradients(x, y)
    activations = x  # in this toy setting treat activations as the inputs

    state.r = refractory_update(state.r, activations.mean(axis=0, keepdims=True).T, grads["w"])
    gate = gating(state.r, kappa=0.25, T=0.05)
    state.e = eligibility_update(state.e, gate, grads["w"], lam=0.8)
    grad_var = np.square(grads["w"])
    state.B = budget_update(state.B, state.e, grad_var, rho=0.8, delta=0.5, xi=0.05)

    # Apply the NIEP-adjusted update to the shadow weights.
    adaptive_lr = state.B * gate
    for name in state.w_tilde:
        state.w_tilde[name] = model.params[name] - adaptive_lr * grads[name]

    # Evaluate both models on a validation split (here: reuse training data).
    def mse(params: Dict[str, np.ndarray]) -> float:
        preds = x @ params["w"] + params["b"]
        return float(np.mean(np.square(preds - y)))

    metrics = {
        "main": {"loss": mse(model.params)},
        "shadow": {"loss": mse(state.w_tilde)},
    }

    updated_params, committed = safe_commit(model.params, state.w_tilde, chi=0.3, validation_metrics=metrics)
    status = "committed" if committed else "rejected"

    print("Validation main loss:", metrics["main"]["loss"])
    print("Validation shadow loss:", metrics["shadow"]["loss"])
    print("Safe commit status:", status)
    print("Updated parameters:\n", updated_params)


if __name__ == "__main__":
    main()
