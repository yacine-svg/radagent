"""B3 tests: the REAL agents through the real MCP servers. Run:  python -m pytest tests/test_real_agents.py -q

- MCP servers run in-process (RADAGENT_MCP=inprocess): same code path as the app, no subprocess.
- Person A's ml/vision.py and ml/text_nlp.py are replaced by small stand-ins with the contract's
  output, so these tests keep working before and after A merges the real modules.
- PubMed is served from a temporary cache (no network).
"""
import copy
import importlib.machinery
import json
import sys
import types

import pytest

pytest.importorskip("mcp.server.mcpserver", reason="MCP SDK v2 not installed")

import ml
from contracts import FinalState, KnowledgeResult
from ml.labels import CLASSES
from servers import knowledge, patients
from agents.common import ROOT
import agents.graph as G

EX = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "examples").glob("case_*.json")}
BY_IMAGE = {ex["image_path"]: ex for ex in EX.values()}
EXPECTED = {"agreement": "ok", "conflict": "ok", "abstain": "abstain"}


def fake_module(name, **funcs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for k, v in funcs.items():
        setattr(mod, k, v)
    return mod


def fake_predict(path):
    ex = BY_IMAGE[path]["image_result"]
    return {"findings": copy.deepcopy(ex["findings"]), "model": ex["model"]}


@pytest.fixture(autouse=True)
def real_agents_inprocess(monkeypatch, tmp_path):
    monkeypatch.setenv("RADAGENT_MCP", "inprocess")
    # stand-in for A's ml/vision.py
    vision = fake_module("ml.vision", predict=fake_predict, gradcam=lambda path, label, out: out)
    monkeypatch.setitem(sys.modules, "ml.vision", vision)
    monkeypatch.setattr(ml, "vision", vision, raising=False)
    # offline PubMed: a cache with 2 refs per class query
    cache = {knowledge.PUBMED_QUERY.format(label=c): [
        {"pmid": f"7{i}{j}", "title": f"Test ref {j} about {c}", "journal": "J", "year": "2020"} for j in (1, 2)]
        for i, c in enumerate(CLASSES)}
    (tmp_path / "pubmed.json").write_text(json.dumps(cache))
    monkeypatch.setattr(knowledge, "PUBMED_CACHE", tmp_path / "pubmed.json")
    monkeypatch.setattr(patients, "DB", tmp_path / "patients.db")
    monkeypatch.setattr(G, "PARALLEL", G.select_agents(fake=False))


def inputs(case):
    return G.example_inputs(case)


def real_trace(final):
    return {t["node"]: t["message"] for t in final["trace"]}


@pytest.mark.parametrize("case", list(G.CASES))
def test_examples_with_real_agents(case):
    assert [fn.__module__ for _, fn in G.PARALLEL] == ["agents.real_agents"] * 3
    final = G.run_sequential(inputs(case))
    FinalState.model_validate(final)
    assert final["decision"] == EXPECTED[case]
    assert not any("FAKE" in m for m in real_trace(final).values())
    assert len(final["knowledge"]["by_class"]) == len(CLASSES)
    assert final["citations"] and all(p.startswith("7") for p in final["citations"])


def test_text_agent_adds_patient_history_from_patient_server():
    final = G.run_sequential(inputs("agreement"))                 # P007 has a history in the db
    tr = final["text_result"]
    assert tr["history_used"] and "congestive heart failure" in tr["analyzed_text"]
    assert final["patient"]["found"] and final["patient"]["age"] == 62
    assert "placeholder" in real_trace(final)["text_agent"]       # A3 not merged in this test


def test_vision_failure_becomes_image_failed(monkeypatch):
    def broken(path):
        raise OSError("cannot read image")
    monkeypatch.setattr(sys.modules["ml.vision"], "predict", broken)
    final = G.run_sequential(inputs("conflict"))
    FinalState.model_validate(final)
    assert final["decision"] == "abstain"
    assert "cannot read image" in final["image_result"]["error"]
    assert final["fused"] is None


def test_uses_A_text_module_when_present(monkeypatch):
    def analyze_text(text, model=None):
        return {"analyzed_text": text, "entities": [], "history_used": False,
                "suspected": {c: "absent" for c in CLASSES}}
    mod = fake_module("ml.text_nlp", analyze_text=analyze_text)
    monkeypatch.setitem(sys.modules, "ml.text_nlp", mod)
    final = G.run_sequential(inputs("agreement"))
    FinalState.model_validate(final)
    assert real_trace(final)["text_agent"].startswith("LLM analysis")
    assert final["text_result"]["history_used"] is True


def test_text_failure_keeps_the_patient(monkeypatch):
    def analyze_text(text, model=None):
        raise TimeoutError("groq timeout")
    monkeypatch.setitem(sys.modules, "ml.text_nlp", fake_module("ml.text_nlp", analyze_text=analyze_text))
    final = G.run_sequential(inputs("agreement"))
    FinalState.model_validate(final)
    assert final["text_result"]["error"] == "TimeoutError: groq timeout"
    assert final["patient"]["found"]
    assert "text_failed" in {r["code"] for r in final["reasons"]}


def test_knowledge_agent_over_real_stdio(monkeypatch):
    """Launches `python -m servers.knowledge_server` as a subprocess, like the app."""
    monkeypatch.delenv("RADAGENT_MCP")
    from agents.real_agents import knowledge_agent
    import asyncio
    out = asyncio.run(knowledge_agent({"case_id": "t", "flags": {"has_text": True, "has_patient": False}}))
    KnowledgeResult.model_validate(out["knowledge"])
    assert out["knowledge"]["error"] is None


@pytest.mark.parametrize("case", list(G.CASES))
def test_real_agents_langgraph(case):
    pytest.importorskip("langgraph")
    final = G.run(inputs(case))
    FinalState.model_validate(final)
    assert final["decision"] == EXPECTED[case]
