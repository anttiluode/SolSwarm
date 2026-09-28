from __future__ import annotations

import numpy as np


def build_transition() -> np.ndarray:
    """Return the fixed 10-state column-stochastic transport matrix."""
    M = np.zeros((10, 10), dtype=float)
    for base, other_hub in ((0, 5), (5, 0)):
        hub = base
        leaves = range(base + 1, base + 5)

        M[hub, hub] = 0.36
        for leaf in leaves:
            M[leaf, hub] = 0.15
        M[other_hub, hub] = 0.04

        for leaf in leaves:
            M[hub, leaf] = 0.54
            M[leaf, leaf] = 0.42
            M[other_hub, leaf] = 0.04
    return M


def positive_operator(
    trace: np.ndarray, alpha: float = 2.0, a_bias: float = 0.08
) -> np.ndarray:
    """Map persistent trace to a positive mutation-selection operator."""
    trace = np.asarray(trace, dtype=float)
    if trace.shape != (10,):
        raise ValueError("trace must have shape (10,)")
    bias = np.zeros(10, dtype=float)
    bias[:5] = a_bias
    fitness = np.exp(bias + alpha * trace)
    return build_transition() @ np.diag(fitness)


def _spectral_radius(block: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(block))))


def module_growth_ratio(L: np.ndarray) -> float:
    """Return rho(B block) / rho(A block)."""
    L = np.asarray(L, dtype=float)
    if L.shape != (10, 10):
        raise ValueError("L must have shape (10, 10)")
    rho_a = _spectral_radius(L[:5, :5])
    rho_b = _spectral_radius(L[5:, 5:])
    return rho_b / rho_a


def evolve(q0: np.ndarray, L: np.ndarray, steps: int) -> np.ndarray:
    """Projectively evolve a nonnegative state under L."""
    q = np.asarray(q0, dtype=float).copy()
    if q.shape != (10,):
        raise ValueError("q0 must have shape (10,)")
    if np.any(q < 0) or q.sum() <= 0:
        raise ValueError("q0 must be nonnegative with positive mass")
    q /= q.sum()
    hist = [q.copy()]
    for _ in range(steps):
        q = L @ q
        total = float(q.sum())
        if total <= 0 or not np.isfinite(total):
            raise ValueError("operator produced invalid total mass")
        q /= total
        hist.append(q.copy())
    return np.asarray(hist)


def spectral_susceptibility(trace: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Finite-difference d log(rho_B/rho_A) / d trace_i."""
    trace = np.asarray(trace, dtype=float)
    base = np.log(module_growth_ratio(positive_operator(trace)))
    out = np.empty(10, dtype=float)
    for i in range(10):
        perturbed = trace.copy()
        perturbed[i] += eps
        value = np.log(module_growth_ratio(positive_operator(perturbed)))
        out[i] = (value - base) / eps
    return out


def matched_write_receipt(write: float = 0.15, steps: int = 20) -> dict:
    """Compare equal writes to the B hub versus one B leaf."""
    q0 = np.full(10, 0.1, dtype=float)
    hub_trace = np.zeros(10, dtype=float)
    leaf_trace = np.zeros(10, dtype=float)
    hub_trace[5] = write
    leaf_trace[6] = write

    hub_L = positive_operator(hub_trace)
    leaf_L = positive_operator(leaf_trace)
    hub_hist = evolve(q0, hub_L, steps)
    leaf_hist = evolve(q0, leaf_L, steps)

    return {
        "same_initial_state": bool(np.allclose(hub_hist[0], leaf_hist[0])),
        "same_total_write": bool(np.isclose(hub_trace.sum(), leaf_trace.sum())),
        "write": float(write),
        "steps": int(steps),
        "initial_B_mass": float(q0[5:].sum()),
        "hub": {
            "growth_ratio_B_over_A": module_growth_ratio(hub_L),
            "future_B_mass": float(hub_hist[-1, 5:].sum()),
        },
        "leaf": {
            "growth_ratio_B_over_A": module_growth_ratio(leaf_L),
            "future_B_mass": float(leaf_hist[-1, 5:].sum()),
        },
    }
