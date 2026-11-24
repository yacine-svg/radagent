"""Graph tests (B) with the FAKE agents. Run:  python -m pytest tests/test_graph.py -q
(The real agents are tested in tests/test_real_agents.py.)

The sequential runner needs no LangGraph, so these run in CI.
The LangGraph tests run when langgraph is installed (on your laptop).
"""
import json
import pytest

from contracts import FinalState, in_uncertain_band
from agents import fake_agents
from agents.graph import CASES, example_inputs, run_sequential

EXPECTED = {"agreement": "ok", "conflict": "ok", "abstain": "abstain"}


@pytest.fixture(autouse=True)
def fake_agents_only(monkeypatch):
    import agents.graph as G
    monkeypatch.setattr(G, "PARALLEL", G.select_agents(fake=True))


def check(final):
    FinalState.model_validate(final)
    return final


@pytest.mark.parametrize("case", list(CASES))
def test_examples_sequential(case):
    final = check(run_sequential(example_inputs(case)))
    assert final["decision"] == EXPECTED[case]
    nodes = [t["node"] for t in final["trace"]]
    assert nodes[0] == "router" and nodes[-1] == "explainer"
    assert set(nodes[1:4]) == {"image_agent", "text_agent", "knowledge_agent"}


def test_conflict_case_warns_and_cites():
    final = check(run_sequential(example_inputs("conflict")))
    assert [r["code"] for r in final["reasons"]] == ["conflict"]
    assert final["citations"] and all(f"PMID: {p}" in final["report_md"] for p in final["citations"])


def test_abstain_case_is_uncertain_conflict():
    final = check(run_sequential(example_inputs("abstain")))
    codes = {r["code"] for r in final["reasons"]}
    assert {"uncertain_conflict", "patient_not_found"} <= codes
    eff = next(f for f in final["fused"]["findings"] if f["conflict"])
    assert in_uncertain_band(eff["p_fused"])


def test_no_text_no_patient_generates_case_id():
    final = check(run_sequential({"image_path": "data/demo/x.png", "text": "  "}))
    assert final["case_id"].startswith("case-")
    assert final["patient"] is None and final["text_result"]["analyzed_text"] == ""
    assert "no_text" in {r["code"] for r in final["reasons"]}


def test_agreement_without_patient_drops_history():
    inp = example_inputs("agreement") | {"patient_id": None}
    final = check(run_sequential(inp))
    assert final["text_result"]["history_used"] is False
    assert "Patient history" not in final["text_result"]["analyzed_text"]


FALLBACKS = {"image_agent": ("image_result", fake_agents._image_failed, "image_failed"),
             "text_agent": ("text_result", fake_agents._text_failed, "text_failed"),
             "knowledge_agent": ("knowledge", fake_agents._knowledge_failed, "knowledge_failed")}


@pytest.mark.parametrize("agent", list(FALLBACKS))
def test_agent_failure_still_returns_valid_state(monkeypatch, agent):
    """AGENTS.md 6.1: an agent crash becomes an `error` + a reason, never an exception."""
    import agents.graph as G
    from agents.common import safe_node
    key, fallback, code = FALLBACKS[agent]

    def crash(state):
        raise RuntimeError("boom")

    monkeypatch.setattr(G, "PARALLEL", [(n, safe_node(n, fallback)(crash) if n == agent else fn)
                                        for n, fn in G.PARALLEL])
    final = check(G.run_sequential(example_inputs("conflict")))
    assert final[key]["error"] == "RuntimeError: boom"
    assert code in {r["code"] for r in final["reasons"]}
    others = {k for a, (k, _, _) in FALLBACKS.items() if a != agent}
    assert all(final[k].get("error") is None for k in others)


# ---- real LangGraph (skipped when langgraph is not installed) ---------------
@pytest.mark.parametrize("case", list(CASES))
def test_examples_langgraph(case):
    pytest.importorskip("langgraph")
    from agents.graph import run
    final = check(run(example_inputs(case)))
    assert final["decision"] == EXPECTED[case]
    seq = run_sequential(example_inputs(case))
    for key in ("fused", "decision", "reasons", "citations"):
        assert json.dumps(final[key], sort_keys=True) == json.dumps(seq[key], sort_keys=True)
