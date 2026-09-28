from __future__ import annotations


def canonicalize(value, digits: int = 8):
    """Round report-only floats recursively while preserving booleans and integers."""
    if isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [canonicalize(item, digits=digits) for item in value]
    if isinstance(value, tuple):
        return [canonicalize(item, digits=digits) for item in value]
    if isinstance(value, dict):
        return {key: canonicalize(item, digits=digits) for key, item in value.items()}
    return value
