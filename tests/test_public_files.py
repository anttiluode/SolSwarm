import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_artifacts_exist_and_describe_all_three_gates():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    receipt = json.loads((ROOT / "receipts" / "latest.json").read_text(encoding="utf-8"))

    assert "Gate A" in readme and "Gate B" in readme and "Gate C" in readme
    assert "Matched write" in page
    assert "Importance-weighted Oja" in page
    assert set(receipt) == {
        "gate_A_world_written_operator",
        "gate_B_importance_weighted_oja",
        "gate_C_adaptive_boundary",
    }
