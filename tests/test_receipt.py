from solswarm.receipt import canonicalize


def test_canonicalize_rounds_nested_floats_without_changing_bools_or_ints():
    value = {
        "x": 1.23456789012345,
        "nested": [0.9473687620914495, True, 3],
    }
    got = canonicalize(value, digits=10)
    assert got == {
        "x": 1.2345678901,
        "nested": [0.9473687621, True, 3],
    }
