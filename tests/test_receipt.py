from solswarm.receipt import canonicalize


def test_canonicalize_rounds_nested_floats_without_changing_bools_or_ints():
    value = {
        "x": 1.23456789012345,
        "nested": [0.9473687620914495, True, 3],
    }
    got = canonicalize(value)
    assert got == {
        "x": 1.23456789,
        "nested": [0.94736876, True, 3],
    }
