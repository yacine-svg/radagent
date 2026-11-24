"""B2 tests: knowledge-server and patient-server. Run:  python -m pytest tests/test_servers.py -q

Logic tests need nothing but pydantic (no network: PubMed is faked).
MCP tests talk to the real servers through the MCP SDK v2 Client (skipped if mcp isn't installed).
"""
import asyncio
import json
import sys

import pytest

from contracts import GuidelineSnippet, ICD10, PatientRecord, Reference
from ml.labels import CLASSES
from servers import knowledge, patients


# ------------------------------------------------------------------ logic: ICD-10
@pytest.mark.parametrize("label", CLASSES)
def test_icd10_every_class(label):
    icd = ICD10.model_validate(knowledge.icd10_lookup(label))
    assert icd.code and icd.label


def test_icd10_unknown_is_null():
    assert knowledge.icd10_lookup("Pneumothorax") == {"condition": "Pneumothorax", "code": None, "label": None}


# ------------------------------------------------------------------ logic: guidelines
@pytest.mark.parametrize("label", CLASSES)
def test_guidelines_retrieve_the_right_note(label):
    snippets = [GuidelineSnippet.model_validate(s) for s in knowledge.guideline_snippets(label, k=2)]
    assert len(snippets) == 2
    slug = label.replace(" ", "_").lower()
    note = (knowledge.DATA / "guidelines" / f"{slug}.md").read_text(encoding="utf-8")
    assert f"source: {snippets[0].source}" in note        # the top hit comes from the class's own note
    assert " ".join(snippets[0].text.split()[:6]) in " ".join(note.split())


def test_guidelines_unknown_condition_never_raises():
    assert isinstance(knowledge.guideline_snippets("xyz", k=2), list)


# ------------------------------------------------------------------ logic: PubMed (faked network)
def fake_fetch(url, params):
    if url.endswith("esearch.fcgi"):
        return {"esearchresult": {"idlist": ["111", "222"]}}
    return {"result": {"111": {"title": "A study", "source": "Radiology", "pubdate": "2021 Mar"},
                       "222": {"title": "Another", "source": "", "pubdate": "Spring"}}}


def broken_fetch(url, params):
    raise ConnectionError("offline")


def test_pubmed_parses_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "PUBMED_CACHE", tmp_path / "pubmed.json")
    monkeypatch.setattr(knowledge.time, "sleep", lambda s: None)
    refs = [Reference.model_validate(r) for r in knowledge.search_pubmed("cardiomegaly", 2, fetch=fake_fetch)]
    assert [r.pmid for r in refs] == ["111", "222"] and refs[0].year == "2021" and refs[1].year == ""
    # offline afterwards: served from the cache
    assert knowledge.search_pubmed("cardiomegaly", 2, fetch=broken_fetch) == [r.model_dump() for r in refs]


def test_pubmed_offline_without_cache_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "PUBMED_CACHE", tmp_path / "pubmed.json")
    assert knowledge.search_pubmed("anything", 3, fetch=broken_fetch) == []


# ------------------------------------------------------------------ logic: patients
def test_patient_found_and_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(patients, "DB", tmp_path / "patients.db")      # db is built on first use
    p = PatientRecord.model_validate(patients.get_patient_history("P007"))
    assert p.found and p.age == 62 and "congestive heart failure (2019)" in p.history
    assert patients.get_patient_history(" P999 ") == {"patient_id": "P999", "found": False}


def test_example_patient_matches_the_database():
    ex = json.loads((knowledge.ROOT / "examples" / "case_agreement.json").read_text(encoding="utf-8"))
    assert patients.get_patient_history(ex["patient_id"]) == ex["patient"]


# ------------------------------------------------------------------ real MCP servers
mcp_v2 = pytest.importorskip("mcp.server.mcpserver", reason="MCP SDK v2 not installed")


def _call(server, tool, args):
    from mcp import Client
    from agents.mcp_client import tool_output

    async def go():
        async with Client(server) as c:
            return tool_output(await c.call_tool(tool, args))
    return asyncio.run(go())


def test_mcp_tool_names_match_the_contract():
    from mcp import Client
    from servers.knowledge_server import mcp as K
    from servers.patient_server import mcp as P

    async def names(server):
        async with Client(server) as c:
            return sorted(t.name for t in (await c.list_tools()).tools)
    assert asyncio.run(names(K)) == ["guideline_snippets", "icd10_lookup", "search_pubmed"]
    assert asyncio.run(names(P)) == ["get_patient_history"]


def test_mcp_knowledge_tools_return_contract_objects():
    from servers.knowledge_server import mcp as K
    ICD10.model_validate(_call(K, "icd10_lookup", {"condition": "Effusion"}))
    snippets = _call(K, "guideline_snippets", {"condition": "Atelectasis", "k": 2})
    assert len(snippets) == 2 and all(GuidelineSnippet.model_validate(s) for s in snippets)


def test_mcp_patient_over_stdio():
    """Launches `python -m servers.patient_server` as a real subprocess, like the app will."""
    from agents.mcp_client import stdio
    rec = PatientRecord.model_validate(_call(stdio("patient"), "get_patient_history", {"patient_id": "P007"}))
    assert rec.found and rec.sex == "F"
