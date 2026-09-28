from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solswarm.adaptive import adaptive_receipt
from solswarm.oja import oja_receipt
from solswarm.operator import matched_write_receipt, spectral_susceptibility
from solswarm.receipt import canonicalize


def build_receipt() -> dict:
    operator = matched_write_receipt(write=0.15, steps=20)
    operator["spectral_susceptibility"] = spectral_susceptibility(np.zeros(10)).tolist()
    return canonicalize({
        "gate_A_world_written_operator": operator,
        "gate_B_importance_weighted_oja": oja_receipt(seeds=32, steps=3000),
        "gate_C_adaptive_boundary": adaptive_receipt(seeds=64, steps=200),
    })


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "receipts" / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_receipt(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
