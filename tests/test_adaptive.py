import numpy as np

from solswarm.adaptive import (
    adaptive_receipt,
    oracle_proposal,
    proposal_from_trace,
    update_running_estimate,
)


def test_adaptive_proposal_is_strictly_positive_and_normalized():
    trace = np.array([0.0, 1.0, 2.0, 0.5, 0.0, 0.2, 3.0, 0.1])
    q = proposal_from_trace(trace, beta=1.5, exploration=0.2)
    assert np.isclose(q.sum(), 1.0)
    assert np.all(q > 0)


def test_running_utility_estimates_magnitude_instead_of_counting_visits():
    estimates = np.zeros(3)
    counts = np.zeros(3, dtype=int)
    for _ in range(20):
        update_running_estimate(estimates, counts, index=1, observation=2.5)
    assert counts[1] == 20
    assert np.isclose(estimates[1], 2.5)
    assert np.isclose(estimates.sum(), 2.5)


def test_oracle_proposal_is_positive_normalized_and_nonuniform():
    q = oracle_proposal()
    assert np.isclose(q.sum(), 1.0)
    assert np.all(q > 0)
    assert q.max() > q.min()


def test_adaptive_receipt_keeps_failed_accumulator_and_reports_new_controls():
    receipt = adaptive_receipt(seeds=64, steps=200)
    assert receipt["adaptive_weighted_median_angle_deg"] < receipt["adaptive_naive_median_angle_deg"]
    assert receipt["adaptive_naive_median_angle_deg"] > 15.0
    assert "estimated_utility_median_angle_deg" in receipt
    assert "oracle_median_angle_deg" in receipt
    assert "estimated_utility_seed_win_fraction" in receipt
    assert "oracle_seed_win_fraction" in receipt
