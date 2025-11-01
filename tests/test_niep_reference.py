import pytest

np = pytest.importorskip("numpy")

from src.python.niep_reference import (
    NIEPState,
    budget_update,
    eligibility_update,
    gating,
    refractory_update,
    safe_commit,
)


def test_refractory_update_limits_growth():
    r = np.zeros((2,))
    activations = np.array([2.0, 0.5])
    grads = np.array([0.1, -0.2])

    updated = refractory_update(r, activations, grads, alpha=0.5, beta=0.2, gamma=0.3, clip=(0.0, 1.0))
    expected = np.clip(0.5 * r + 0.2 * np.abs(activations) + 0.3 * np.abs(grads), 0.0, 1.0)
    np.testing.assert_allclose(updated, expected)


def test_gating_behaviour():
    r = np.array([0.0, 0.3, 0.6])
    gate = gating(r, kappa=0.3, T=0.1)

    assert np.all(gate >= 0.0)
    assert np.all(gate <= 1.0)
    assert gate[0] > gate[1] > gate[2]


def test_eligibility_update_uses_gate():
    e = np.zeros(3)
    gate = np.array([1.0, 0.5, 0.0])
    grads = np.array([1.0, 1.0, 1.0])

    updated = eligibility_update(e, gate, grads, lam=0.5)
    np.testing.assert_allclose(updated, np.array([0.5, 0.25, 0.0]))


def test_budget_update_applies_floor():
    B = np.zeros(3)
    e = np.array([0.0, 1.0, 2.0])
    grad_var = np.array([1e-6, 0.1, 0.5])

    updated = budget_update(B, e, grad_var, rho=0.5, delta=1.0, xi=0.2)
    assert np.all(updated >= 0.2)


def test_safe_commit_requires_improvement():
    params = {"w": np.array([0.0, 0.0])}
    shadow = {"w": np.array([1.0, 1.0])}

    metrics = {
        "main": {"loss": 0.5, "accuracy": 0.8},
        "shadow": {"loss": 0.4, "accuracy": 0.85},
    }
    updated, committed = safe_commit(params.copy(), shadow, chi=0.5, validation_metrics=metrics)
    assert committed
    np.testing.assert_allclose(updated["w"], np.array([0.5, 0.5]))

    metrics_bad = {
        "main": {"loss": 0.5, "accuracy": 0.8},
        "shadow": {"loss": 0.6, "accuracy": 0.85},
    }
    updated_bad, committed_bad = safe_commit(params.copy(), shadow, chi=0.5, validation_metrics=metrics_bad)
    assert not committed_bad
    np.testing.assert_allclose(updated_bad["w"], params["w"])


def test_safe_commit_without_metrics_always_commits():
    params = {"w": np.array([0.0, 0.0])}
    shadow = {"w": np.array([1.0, 1.0])}

    updated, committed = safe_commit(params, shadow, chi=1.0)
    assert committed
    np.testing.assert_allclose(updated["w"], shadow["w"])


def test_state_dataclass_allows_mutation():
    params = {"w": np.array([0.0])}
    state = NIEPState(r=np.zeros(1), e=np.zeros(1), B=np.ones(1), w_tilde=params)

    state.w_tilde["w"] += 1.0
    np.testing.assert_allclose(state.w_tilde["w"], np.array([1.0]))
