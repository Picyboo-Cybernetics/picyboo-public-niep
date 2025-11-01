"""Neuro-Inspired Error Propagation (NIEP) reference utilities.

The implementations below follow the update logic introduced in the
"Neuro-Inspired Error Propagation (NIEP)" whitepaper.  They are intentionally
framework agnostic so the routines can be embedded in PyTorch, TensorFlow or
NumPy based codebases.

The module exposes lightweight state containers and numerical helpers for the
four core mechanisms described in the paper:

* Refractory gating that temporarily suppresses large weight updates after a
  parameter experienced a strong activation or gradient spike.
* Eligibility traces that accumulate gated gradients over time to enable
  delayed credit assignment.
* Error budgets that track how much additional update magnitude can be safely
  applied without destabilising the model.
* A safe-commit routine that softly merges shadow weights into the main model
  only if validation metrics indicate an improvement.

The goal is to provide a faithful and well documented translation of the
pseudocode from Sections 4 and 5 of the paper so that practitioners can use the
functions as building blocks for their own continual-learning loops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

import numpy as np

ArrayLike = np.ndarray


@dataclass
class NIEPState:
    """Container holding the per-parameter NIEP statistics.

    Attributes
    ----------
    r:
        Refractory state for each parameter.  Values increase when the
        associated activation or gradient is large and gradually decay back to
        zero through :func:`refractory_update`.
    e:
        Eligibility traces capturing the gated gradient accumulation.  They are
        updated with :func:`eligibility_update` and later used as inputs when
        computing the effective weight delta.
    B:
        Error budgets representing how much cumulative update magnitude can be
        applied before triggering stabilisation mechanisms.
    w_tilde:
        Shadow weights that receive tentative updates before
        :func:`safe_commit` merges them into the main parameters.
    """

    r: ArrayLike
    e: ArrayLike
    B: ArrayLike
    w_tilde: MutableMapping[str, ArrayLike]


def step_forward_backward(batch: Any, params: Mapping[str, ArrayLike]) -> Tuple[Any, Dict[str, ArrayLike]]:
    """Framework placeholder for the forward/backward passes.

    The whitepaper intentionally leaves the deep-learning framework unspecified.
    This helper mirrors the pseudocode signature and documents the expected
    behaviour: perform a forward pass, compute the loss, run backpropagation and
    return both model activations and gradients.

    Parameters
    ----------
    batch:
        Mini-batch sampled from the current data stream.
    params:
        Mapping of parameter names to tensors/arrays.

    Returns
    -------
    Tuple[Any, Dict[str, ArrayLike]]
        A tuple containing the framework specific activation information and a
        dictionary with gradients keyed by the parameter names.
    """
    raise NotImplementedError("Integrate with your deep-learning framework of choice.")


def refractory_update(
    r: ArrayLike,
    activations: ArrayLike,
    grads: ArrayLike,
    *,
    alpha: float = 0.9,
    beta: float = 0.05,
    gamma: float = 0.05,
    clip: Tuple[float, float] | None = (0.0, 10.0),
) -> ArrayLike:
    """Update refractory states as described in Section 4.1 of the paper.

    The refractory trace follows a leaky integration of absolute activations
    and gradients.  Large spikes cause the refractory value to increase, which
    later attenuates parameter updates through the gating function.  The decay
    factor :math:`\alpha` keeps the state bounded.

    Parameters
    ----------
    r:
        Current refractory state array.
    activations:
        Absolute activations (or post-synaptic responses) associated with the
        parameter.
    grads:
        Raw gradient tensor for the same parameter.
    alpha, beta, gamma:
        Hyperparameters controlling the decay (:math:`\alpha`) and the
        contribution from activations (:math:`\beta`) and gradients
        (:math:`\gamma`).  Defaults match the initialisation table in the
        whitepaper.
    clip:
        Optional ``(min, max)`` tuple that bounds the refractory value to avoid
        numerical overflow.

    Returns
    -------
    numpy.ndarray
        Updated refractory state.
    """

    r = alpha * r + beta * np.abs(activations) + gamma * np.abs(grads)
    if clip is not None:
        r = np.clip(r, clip[0], clip[1])
    return r


def gating(r: ArrayLike, *, kappa: float = 0.3, T: float = 0.05) -> ArrayLike:
    """Compute the refractory gate for each parameter.

    The gate follows a smooth logistic shape ``sigmoid((kappa - r) / T)``.
    When the refractory trace ``r`` is large (recent intense activity) the gate
    tends towards zero, effectively freezing the parameter.  As ``r`` relaxes,
    the value approaches one which enables regular learning again.

    Parameters
    ----------
    r:
        Refractory state array.
    kappa:
        Set-point controlling the refractory level at which learning is
        half-suppressed.
    T:
        Temperature of the logistic curve.  Lower values produce sharper
        transitions.  The implementation guards against zero to avoid division
        by zero.

    Returns
    -------
    numpy.ndarray
        Element-wise gate between ``0`` and ``1``.
    """

    T = max(T, 1e-6)
    gate = 1.0 / (1.0 + np.exp(-(kappa - r) / T))
    return gate


def eligibility_update(
    e: ArrayLike,
    gate: ArrayLike,
    grads: ArrayLike,
    *,
    lam: float = 0.9,
) -> ArrayLike:
    """Update eligibility traces using leaky integration.

    The update implements ``e <- lambda * e + (1 - lambda) * gate * grads``.
    Gates close to zero effectively pause the trace, while sustained gradients
    gradually accumulate which supports delayed credit assignment.

    Parameters
    ----------
    e:
        Current eligibility trace values.
    gate:
        Output from :func:`gating`.
    grads:
        Raw gradients from backpropagation.
    lam:
        Decay factor :math:`\lambda`.  Values close to ``1`` preserve more
        history; smaller values react faster to new information.

    Returns
    -------
    numpy.ndarray
        Updated eligibility traces.
    """

    lam = np.clip(lam, 0.0, 1.0)
    return lam * e + (1.0 - lam) * gate * grads


def budget_update(
    B: ArrayLike,
    e: ArrayLike,
    grad_var: ArrayLike,
    *,
    rho: float = 0.9,
    delta: float = 1.0,
    xi: float = 0.1,
    eps: float = 1e-12,
) -> ArrayLike:
    """Variance-aware error budget update.

    The whitepaper proposes budgeting updates in proportion to the recent
    eligibility magnitude while penalising volatile gradients.  This reference
    implementation keeps an exponential moving average (``rho``) of the
    previous budget and adds a scaled contribution from the current eligibility
    signal.  High gradient variance reduces the added amount which mirrors the
    stability objective.

    Parameters
    ----------
    B:
        Previous budget values.
    e:
        Eligibility traces.
    grad_var:
        Running variance (or second moment) estimate of the gradients.
    rho:
        Decay applied to the previous budget.
    delta:
        Scale factor translating eligibility mass into budget increase.
    xi:
        Minimal budget floor ensuring that each parameter retains some learning
        capacity even in the presence of large variance.
    eps:
        Numerical constant preventing division by zero.

    Returns
    -------
    numpy.ndarray
        Updated budgets with a lower bound of ``xi``.
    """

    scaled = delta * np.abs(e) / (np.sqrt(grad_var) + eps)
    B = rho * B + (1.0 - rho) * scaled
    return np.maximum(B, xi)


def _blend_params(
    params: MutableMapping[str, ArrayLike],
    w_tilde: Mapping[str, ArrayLike],
    chi: float,
) -> MutableMapping[str, ArrayLike]:
    """Internal helper that performs the convex merge of parameters."""

    for name, tensor in params.items():
        if name not in w_tilde:
            raise KeyError(f"Shadow weights missing parameter '{name}'.")
        params[name] = (1.0 - chi) * tensor + chi * w_tilde[name]
    return params


def safe_commit(
    params: MutableMapping[str, ArrayLike],
    w_tilde: Mapping[str, ArrayLike],
    *,
    chi: float = 0.25,
    validation_metrics: Mapping[str, float] | None = None,
    prefer_lower: Iterable[str] | None = ("loss",),
) -> Tuple[MutableMapping[str, ArrayLike], bool]:
    """Merge shadow weights when validation metrics indicate an improvement.

    The *safe-commit* protocol decouples tentative learning (performed on the
    shadow weights ``w_tilde``) from the production parameters ``params``.  Only
    when the validation diagnostics confirm progress are the two sets blended.

    Parameters
    ----------
    params:
        Mutable mapping containing the live model parameters.  The function
        updates the mapping in place for convenience and returns it as well.
    w_tilde:
        Shadow weights that have been updated using NIEP's gated gradients.
    chi:
        Blending factor.  A value of ``0.25`` means that 25% of the shadow
        update is injected into the live model upon a successful commit.
    validation_metrics:
        Optional mapping with at least the keys ``"main"`` and ``"shadow"`` or
        any other metric names.  When provided, the commit is accepted iff the
        shadow metric improves over the main metric for all monitored entries.
        When omitted the function assumes validation succeeded and always
        commits.
    prefer_lower:
        Iterable listing the metric names where a *lower* value is considered an
        improvement (e.g. losses).  Metrics not in this iterable are treated as
        "higher is better" (e.g. accuracy).

    Returns
    -------
    Tuple[MutableMapping[str, ArrayLike], bool]
        Updated parameters and a boolean indicating whether a commit happened.
    """

    if validation_metrics is None:
        return _blend_params(params, w_tilde, chi), True

    if "main" not in validation_metrics or "shadow" not in validation_metrics:
        raise KeyError("validation_metrics must contain 'main' and 'shadow'.")

    main_metrics = validation_metrics["main"]
    shadow_metrics = validation_metrics["shadow"]

    if not isinstance(main_metrics, Mapping) or not isinstance(shadow_metrics, Mapping):
        raise TypeError("validation_metrics['main'] and ['shadow'] must be mappings of metric names to values.")

    prefer_lower = set(prefer_lower or ())
    commit = True
    for metric, main_val in main_metrics.items():
        if metric not in shadow_metrics:
            raise KeyError(f"Shadow metrics missing entry '{metric}'.")
        shadow_val = shadow_metrics[metric]
        if metric in prefer_lower:
            if shadow_val >= main_val:
                commit = False
                break
        else:
            if shadow_val <= main_val:
                commit = False
                break

    if commit:
        params = _blend_params(params, w_tilde, chi)
    return params, commit


__all__ = [
    "ArrayLike",
    "NIEPState",
    "step_forward_backward",
    "refractory_update",
    "gating",
    "eligibility_update",
    "budget_update",
    "safe_commit",
]
