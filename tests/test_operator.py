import numpy as np

from solswarm.operator import (
    analytic_spectral_susceptibility,
    build_transition,
    matched_write_receipt,
    spectral_susceptibility,
)


def test_transition_is_column_stochastic():
    M = build_transition()
    assert M.shape == (10, 10)
    assert np.all(M >= 0)
    assert np.allclose(M.sum(axis=0), 1.0)


def test_same_write_different_address_flips_only_hub_mode():
    receipt = matched_write_receipt(write=0.15, steps=20)
    assert receipt["same_initial_state"] is True
    assert receipt["same_total_write"] is True
    assert receipt["hub"]["growth_ratio_B_over_A"] > 1.0
    assert receipt["leaf"]["growth_ratio_B_over_A"] < 1.0
    assert receipt["hub"]["future_B_mass"] - receipt["leaf"]["future_B_mass"] > 0.15


def test_hub_has_greater_spectral_susceptibility_than_leaves():
    sens = spectral_susceptibility(np.zeros(10))
    b = sens[5:]
    assert b[0] > max(b[1:])
    assert b[0] > 1.5 * np.mean(b[1:])


def test_analytic_perron_elasticity_matches_finite_difference_susceptibility():
    trace = np.zeros(10)
    analytic = analytic_spectral_susceptibility(trace, alpha=2.0)
    finite = spectral_susceptibility(trace, eps=1e-6)
    assert np.allclose(analytic, finite, atol=2e-6, rtol=0.0)
    assert np.isclose(-analytic[:5].sum(), 2.0, atol=1e-10)
    assert np.isclose(analytic[5:].sum(), 2.0, atol=1e-10)
