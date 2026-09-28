import numpy as np

from solswarm.adaptive import adaptive_receipt, proposal_from_trace


def test_adaptive_proposal_is_strictly_positive_and_normalized():
    trace = np.array([0.0, 1.0, 2.0, 0.5, 0.0, 0.2, 3.0, 0.1])
    q = proposal_from_trace(trace, beta=1.5, exploration=0.2)
    assert np.isclose(q.sum(), 1.0)
    assert np.all(q > 0)


def test_importance_correction_restores_mode_but_does_not_claim_efficiency_win():
    receipt = adaptive_receipt(seeds=64, steps=200)
    assert receipt["adaptive_weighted_median_angle_deg"] < receipt["adaptive_naive_median_angle_deg"]
    assert receipt["adaptive_weighted_median_angle_deg"] < 15.0
    assert receipt["adaptive_naive_median_angle_deg"] > 15.0
    assert receipt["weighted_adaptive_beats_uniform_median"] is False
