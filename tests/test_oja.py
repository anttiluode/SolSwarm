import numpy as np

from solswarm.oja import error_proposal, feature_world, oja_receipt, principal_direction


def test_target_mode_is_horizontal_and_error_proposal_is_strongly_vertical_biased():
    features, p, error = feature_world()
    target = principal_direction(features, p)
    q = error_proposal(error, beta=3.0)
    assert abs(target[0]) > 0.9
    assert q[-2:].sum() > 0.75
    assert np.isclose(q.sum(), 1.0)


def test_importance_weighted_oja_recovers_world_mode_under_biased_scouts():
    receipt = oja_receipt(seeds=32, steps=3000)
    assert receipt["weighted_median_angle_deg"] < 8.0
    assert receipt["naive_median_angle_deg"] > 60.0
    assert receipt["weighted_median_angle_deg"] < receipt["naive_median_angle_deg"]
