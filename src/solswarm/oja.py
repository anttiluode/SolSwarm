from __future__ import annotations

import numpy as np


def feature_world() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed site features, uniform target p, and a misleading error field."""
    features = np.array(
        [
            [2.2, 0.2],
            [-2.1, -0.1],
            [1.8, -0.3],
            [-1.9, 0.25],
            [2.0, 0.4],
            [-2.2, -0.35],
            [0.2, 2.6],
            [-0.15, -2.7],
        ],
        dtype=float,
    )
    p = np.full(len(features), 1.0 / len(features), dtype=float)
    error = np.array([0.1] * 6 + [1.0, 1.0], dtype=float)
    return features, p, error


def principal_direction(features: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Principal eigenvector of the probability-weighted second moment."""
    features = np.asarray(features, dtype=float)
    probs = np.asarray(probs, dtype=float)
    C = sum(float(probs[i]) * np.outer(features[i], features[i]) for i in range(len(features)))
    values, vectors = np.linalg.eigh(C)
    v = vectors[:, int(np.argmax(values))]
    if v[0] < 0:
        v = -v
    return v / np.linalg.norm(v)


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    ex = np.exp(shifted)
    return ex / ex.sum()


def error_proposal(error: np.ndarray | None = None, beta: float = 3.0) -> np.ndarray:
    if error is None:
        _, _, error = feature_world()
    return _softmax(beta * np.asarray(error, dtype=float))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Sign-invariant angle between one-dimensional subspaces."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cosine = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def run_oja(
    seed: int,
    steps: int = 3000,
    weighted: bool = False,
    proposal: np.ndarray | None = None,
) -> np.ndarray:
    """Run normalized Oja updates under a fixed proposal."""
    features, p, error = feature_world()
    q = error_proposal(error) if proposal is None else np.asarray(proposal, dtype=float)
    if q.shape != p.shape or np.any(q <= 0) or not np.isclose(q.sum(), 1.0):
        raise ValueError("proposal must be strictly positive and sum to one")

    rng = np.random.default_rng(seed)
    w = np.array([1.0, 1.0], dtype=float)
    w /= np.linalg.norm(w)

    for t in range(steps):
        i = int(rng.choice(len(features), p=q))
        z = features[i]
        importance = float(p[i] / q[i]) if weighted else 1.0
        eta = 0.03 / np.sqrt(1.0 + t / 200.0)
        y = float(w @ z)
        w = w + eta * importance * y * (z - y * w)
        norm = float(np.linalg.norm(w))
        if not np.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Oja iterate became invalid")
        w /= norm
    return w


def oja_receipt(seeds: int = 32, steps: int = 3000) -> dict:
    features, p, error = feature_world()
    q = error_proposal(error)
    target = principal_direction(features, p)

    naive = []
    weighted = []
    for seed in range(seeds):
        naive.append(angle_deg(run_oja(seed, steps=steps, weighted=False, proposal=q), target))
        weighted.append(angle_deg(run_oja(seed, steps=steps, weighted=True, proposal=q), target))

    return {
        "seeds": int(seeds),
        "steps": int(steps),
        "proposal_mass_on_high_error_sites": float(q[-2:].sum()),
        "target_direction": target.tolist(),
        "naive_median_angle_deg": float(np.median(naive)),
        "weighted_median_angle_deg": float(np.median(weighted)),
        "weighted_max_angle_deg": float(np.max(weighted)),
    }
