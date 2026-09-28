from __future__ import annotations

import numpy as np

from .oja import angle_deg, feature_world, principal_direction


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    ex = np.exp(values)
    return ex / ex.sum()


def proposal_from_trace(
    trace: np.ndarray, beta: float = 1.5, exploration: float = 0.2
) -> np.ndarray:
    """Trace-biased proposal with a uniform exploration floor."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    if not (0.0 < exploration <= 1.0):
        raise ValueError("exploration must be in (0, 1]")
    soft = _softmax(beta * trace)
    uniform = np.full(trace.shape, 1.0 / len(trace), dtype=float)
    return (1.0 - exploration) * soft + exploration * uniform


def _initial_mode() -> np.ndarray:
    w = np.array([1.0, 1.0], dtype=float)
    return w / np.linalg.norm(w)


def run_uniform(seed: int, steps: int = 200) -> float:
    features, p, _ = feature_world()
    target = principal_direction(features, p)
    rng = np.random.default_rng(seed)
    w = _initial_mode()

    for t in range(steps):
        i = int(rng.choice(len(features), p=p))
        z = features[i]
        y = float(w @ z)
        eta = 0.03 / np.sqrt(1.0 + t / 100.0)
        w = w + eta * y * (z - y * w)
        w /= np.linalg.norm(w)
    return angle_deg(w, target)


def run_adaptive(
    seed: int,
    steps: int = 200,
    weighted: bool = True,
    beta: float = 1.5,
    decay: float = 0.97,
    exploration: float = 0.2,
) -> float:
    """Closed-loop scout: update-magnitude writes trace; trace biases later sampling."""
    features, p, _ = feature_world()
    target = principal_direction(features, p)
    rng = np.random.default_rng(seed)
    w = _initial_mode()
    trace = np.zeros(len(features), dtype=float)

    for t in range(steps):
        q = proposal_from_trace(trace, beta=beta, exploration=exploration)
        i = int(rng.choice(len(features), p=q))
        z = features[i]
        y = float(w @ z)
        raw_update = y * (z - y * w)
        importance = float(p[i] / q[i]) if weighted else 1.0
        eta = 0.03 / np.sqrt(1.0 + t / 100.0)
        w = w + eta * importance * raw_update
        w /= np.linalg.norm(w)

        trace *= decay
        trace[i] += float(np.linalg.norm(raw_update))

    return angle_deg(w, target)


def adaptive_receipt(seeds: int = 64, steps: int = 200) -> dict:
    uniform = np.array([run_uniform(seed, steps=steps) for seed in range(seeds)])
    weighted = np.array(
        [run_adaptive(seed, steps=steps, weighted=True) for seed in range(seeds)]
    )
    naive = np.array(
        [run_adaptive(seed, steps=steps, weighted=False) for seed in range(seeds)]
    )

    uniform_median = float(np.median(uniform))
    weighted_median = float(np.median(weighted))
    naive_median = float(np.median(naive))
    return {
        "seeds": int(seeds),
        "steps": int(steps),
        "uniform_median_angle_deg": uniform_median,
        "adaptive_weighted_median_angle_deg": weighted_median,
        "adaptive_naive_median_angle_deg": naive_median,
        "weighted_adaptive_beats_uniform_median": bool(weighted_median < uniform_median),
        "weighted_adaptive_seed_win_fraction": float(np.mean(weighted < uniform)),
    }
