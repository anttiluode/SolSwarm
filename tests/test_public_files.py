import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_artifacts_exist_and_describe_all_three_gates():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    page = (ROOT / "index.html").read_text(encoding="utf-8")
    receipt = json.loads((ROOT / "receipts" / "latest.json").read_text(encoding="utf-8"))

    assert "Gate A" in readme and "Gate B" in readme and "Gate C" in readme
    assert "SolSwarm Live" in page
    assert "Gate B: the swarm's view of the world vs the world's own" in page
    assert "Gate C: three samplers, same number of looks" in page
    assert "Gate A: where a write would change the future most" in page
    assert "u·v elasticity" in page
    assert set(receipt) == {
        "gate_A_world_written_operator",
        "gate_B_importance_weighted_oja",
        "gate_C_adaptive_boundary",
    }
