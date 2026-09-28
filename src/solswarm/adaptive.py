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
    """Original v0 trace-softmax proposal with a uniform exploration floor."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    if not (0.0 < exploration <= 1.0):
        raise ValueError("exploration must be in (0, 1]")
    soft = _softmax(beta * trace)
    uniform = np.full(trace.shape, 1.0 / len(trace), dtype=float)
    return (1.0 - exploration) * soft + exploration * uniform


def proposal_from_utility(
    utility: np.ndarray, target_probs: np.ndarray, exploration: float = 0.2
) -> np.ndarray:
    """Proposal proportional to p_i * estimated ||g_i|| plus exploration.

    For a fixed stochastic-gradient state, q_i proportional to p_i ||g_i||
    is the standard variance-minimizing importance proposal. The exploration
    mixture keeps all target-supported sites reachable while utility is learned.
    """
    utility = np.asarray(utility, dtype=float)
    target_probs = np.asarray(target_probs, dtype=float)
    if utility.shape != target_probs.shape or utility.ndim != 1:
        raise ValueError("utility and target_probs must be matching vectors")
    if np.any(utility < 0) or np.any(target_probs <= 0):
        raise ValueError("utility must be nonnegative and target_probs positive")
    if not np.isclose(target_probs.sum(), 1.0):
        raise ValueError("target_probs must sum to one")
    if not (0.0 < exploration <= 1.0):
        raise ValueError("exploration must be in (0, 1]")

    weighted = target_probs * utility
    if float(weighted.sum()) <= 0:
        exploit = target_probs.copy()
    else:
        exploit = weighted / weighted.sum()
    return (1.0 - exploration) * exploit + exploration * target_probs


def update_running_estimate(
    estimates: np.ndarray,
    counts: np.ndarray,
    index: int,
    observation: float,
) -> None:
    """Update one site's arithmetic-mean utility estimate in place."""
    if observation < 0 or not np.isfinite(observation):
        raise ValueError("observation must be finite and nonnegative")
    counts[index] += 1
    estimates[index] += (observation - estimates[index]) / float(counts[index])


def _initial_mode() -> np.ndarray:
    w = np.array([1.0, 1.0], dtype=float)
    return w / np.linalg.norm(w)


def _raw_update(features: np.ndarray, mode: np.ndarray, index: int) -> np.ndarray:
    z = features[index]
    y = float(mode @ z)
    return y * (z - y * mode)


def oracle_proposal(mode: np.ndarray | None = None, exploration: float = 0.2) -> np.ndarray:
    """Variance-aware proposal with privileged access to every site's update norm.

    When ``mode`` is omitted, the true target principal direction is used only
    to expose the proposal as a deterministic diagnostic. ``run_oracle`` passes
    the learner's current mode, so the oracle recomputes q_i ∝ p_i ||g_i(w)||
    at every observation.
    """
    features, p, _ = feature_world()
    if mode is None:
        mode = principal_direction(features, p)
    mode = np.asarray(mode, dtype=float)
    mode = mode / np.linalg.norm(mode)
    utility = np.array(
        [np.linalg.norm(_raw_update(features, mode, i)) for i in range(len(features))],
        dtype=float,
    )
    return proposal_from_utility(utility, p, exploration=exploration)


def run_uniform(seed: int, steps: int = 200) -> float:
    features, p, _ = feature_world()
    target = principal_direction(features, p)
    rng = np.random.default_rng(seed)
    w = _initial_mode()

    for t in range(steps):
        i = int(rng.choice(len(features), p=p))
        raw_update = _raw_update(features, w, i)
        eta = 0.03 / np.sqrt(1.0 + t / 100.0)
        w = w + eta * raw_update
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
    """Original v0 accumulator: visited update magnitudes build attractive trace.

    This deliberately remains as the failed rich-get-richer control. Because
    every visit adds to trace, trace conflates per-site utility with visit count.
    """
    features, p, _ = feature_world()
    target = principal_direction(features, p)
    rng = np.random.default_rng(seed)
    w = _initial_mode()
    trace = np.zeros(len(features), dtype=float)

    for t in range(steps):
        q = proposal_from_trace(trace, beta=beta, exploration=exploration)
        i = int(rng.choice(len(features), p=q))
        raw_update = _raw_update(features, w, i)
        importance = float(p[i] / q[i]) if weighted else 1.0
        eta = 0.03 / np.sqrt(1.0 + t / 100.0)
        w = w + eta * importance * raw_update
        w /= np.linalg.norm(w)

        trace *= decay
        trace[i] += float(np.linalg.norm(raw_update))

    return angle_deg(w, target)


def run_estimated_utility(
    seed: int,
    steps: int = 200,
    exploration: float = 0.2,
    prior_utility: float = 1.0,
    prior_count: int = 1,
) -> float:
    """Adaptive sampler that estimates per-site update magnitude instead of visits."""
    features, p, _ = feature_world()
    target = principal_direction(features, p)
    rng = np.random.default_rng(seed)
    w = _initial_mode()
    estimates = np.full(len(features), prior_utility, dtype=float)
    counts = np.full(len(features), prior_count, dtype=int)

    for t in range(steps):
        q = proposal_from_utility(estimates, p, exploration=exploration)
        i = int(rng.choice(len(features), p=q))
        raw_update = _raw_update(features, w, i)
        importance = float(p[i] / q[i])
        eta = 0.03 / np.sqrt(1.0 + t / 100.0)
        w = w + eta * importance * raw_update
        w /= np.linalg.norm(w)
        update_running_estimate(
            estimates,
            counts,
            index=i,
            observation=float(np.linalg.norm(raw_update)),
        )

    return angle_deg(w, target)


def run_oracle(seed: int, steps: int = 200, exploration: float = 0.2) -> float:
    """Privileged control using all current per-site update norms each step."""
    features, p, _ = feature_world()
    target = principal_direction(features, p)
    rng = np.random.default_rng(seed)
    w = _initial_mode()

    for t in range(steps):
        q = oracle_proposal(w, exploration=exploration)
        i = int(rng.choice(len(features), p=q))
        raw_update = _raw_update(features, w, i)
        importance = float(p[i] / q[i])
        eta = 0.03 / np.sqrt(1.0 + t / 100.0)
        w = w + eta * importance * raw_update
        w /= np.linalg.norm(w)

    return angle_deg(w, target)


def adaptive_receipt(seeds: int = 64, steps: int = 200) -> dict:
    uniform = np.array([run_uniform(seed, steps=steps) for seed in range(seeds)])
    accumulating_weighted = np.array(
        [run_adaptive(seed, steps=steps, weighted=True) for seed in range(seeds)]
    )
    accumulating_naive = np.array(
        [run_adaptive(seed, steps=steps, weighted=False) for seed in range(seeds)]
    )
    estimated = np.array(
        [run_estimated_utility(seed, steps=steps) for seed in range(seeds)]
    )
    oracle = np.array([run_oracle(seed, steps=steps) for seed in range(seeds)])

    uniform_median = float(np.median(uniform))
    accumulating_weighted_median = float(np.median(accumulating_weighted))
    accumulating_naive_median = float(np.median(accumulating_naive))
    estimated_median = float(np.median(estimated))
    oracle_median = float(np.median(oracle))
    return {
        "seeds": int(seeds),
        "steps": int(steps),
        "uniform_median_angle_deg": uniform_median,
        # Legacy v0 field names are retained so old receipts/readers remain legible.
        "adaptive_weighted_median_angle_deg": accumulating_weighted_median,
        "adaptive_naive_median_angle_deg": accumulating_naive_median,
        "weighted_adaptive_beats_uniform_median": bool(
            accumulating_weighted_median < uniform_median
        ),
        "weighted_adaptive_seed_win_fraction": float(
            np.mean(accumulating_weighted < uniform)
        ),
        # Explicit names for the corrected interpretation.
        "accumulating_weighted_median_angle_deg": accumulating_weighted_median,
        "accumulating_naive_median_angle_deg": accumulating_naive_median,
        "estimated_utility_median_angle_deg": estimated_median,
        "estimated_utility_beats_uniform_median": bool(estimated_median < uniform_median),
        "estimated_utility_seed_win_fraction": float(np.mean(estimated < uniform)),
        "oracle_median_angle_deg": oracle_median,
        "oracle_beats_uniform_median": bool(oracle_median < uniform_median),
        "oracle_seed_win_fraction": float(np.mean(oracle < uniform)),
    }
