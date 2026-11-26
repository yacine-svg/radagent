"""B6 tests: LLM explainer guards (fake Groq client) + app render helpers. No network, no Streamlit."""
import types

import pytest

import agents.explainer as E
import agents.graph as G
from app.render import cited_refs, decision_banner, finding_label, highlight_html, report_for_display, trace_line
from contracts import DISCLAIMER, REPORT_SECTIONS, FinalState


@pytest.fixture(autouse=True)
def fake_agents_only(monkeypatch):
    monkeypatch.setattr(G, "PARALLEL", G.select_agents(fake=True))


def fake_client(content=None, error=None):
    def create(**kwargs):
        if error:
            raise error
        assert kwargs["temperature"] == 0 and "JSON" in kwargs["messages"][0]["content"]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


def good_report(pmids, extra=""):
    body = {h: "texte" for h in REPORT_SECTIONS}
    body["## Références"] = "\n".join(f"- PMID: {p}" for p in pmids) + extra
    body["## Avertissement"] = DISCLAIMER
    return "\n\n".join(f"{h}\n{body[h]}" for h in REPORT_SECTIONS)


def run_with(monkeypatch, client, case="conflict"):
    monkeypatch.setattr(E, "make_llm", lambda: client)
    final = G.run_sequential(G.example_inputs(case))
    FinalState.model_validate(final)
    return final, final["trace"][-1]["message"]


def test_no_key_uses_template(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    final = G.run_sequential(G.example_inputs("conflict"))
    assert "(template)" in final["trace"][-1]["message"]


def test_llm_report_is_used_when_grounded(monkeypatch):
    final, msg = run_with(monkeypatch, fake_client(good_report(["91000001"])))
    assert "(LLM)" in msg and final["citations"] == ["91000001"]


def test_unknown_pmid_is_removed(monkeypatch):
    final, msg = run_with(monkeypatch, fake_client(good_report(["91000001"], "\n- PMID: 12345678 invented")))
    assert "(LLM)" in msg and "12345678" not in final["report_md"] and final["citations"] == ["91000001"]


def test_pmid_format_is_normalised(monkeypatch):
    final, _ = run_with(monkeypatch, fake_client(good_report([]).replace("## Références\n", "## Références\n- PMID 91000001\n")))
    assert final["citations"] == ["91000001"]


def test_missing_heading_falls_back_to_template(monkeypatch):
    final, msg = run_with(monkeypatch, fake_client(good_report(["91000001"]).replace("## Points de vigilance", "## Notes")))
    assert "LLM fallback: GroundingError" in msg


def test_disclaimer_is_enforced(monkeypatch):
    final, _ = run_with(monkeypatch, fake_client(good_report([]).replace(DISCLAIMER, "Pas de souci.")))
    assert final["report_md"].rstrip().endswith(DISCLAIMER) and "Pas de souci" not in final["report_md"]


def test_llm_error_falls_back_to_template(monkeypatch):
    final, msg = run_with(monkeypatch, fake_client(error=TimeoutError("slow")))
    assert "LLM fallback: TimeoutError" in msg


def test_evidence_only_contains_relevant_references():
    final = G.run_sequential(G.example_inputs("conflict"))
    ev = E.evidence(final)
    assert {r["about"] for r in ev["references"]} == {"Effusion"}
    assert set(ev["icd10"]) == {"Effusion"} and ev["decision"] == "ok"


# ------------------------------------------------------------------ render helpers
def test_highlight_escapes_and_marks():
    html = highlight_html("<b>no</b> effusion, cough", [
        {"text": "effusion", "type": "finding", "negated": True, "span": [10, 18]},
        {"text": "cough", "type": "symptom", "negated": False, "span": [20, 25]}])
    assert "&lt;b&gt;" in html and "<s>effusion</s>" in html and html.count("<mark") == 2


def test_render_helpers_on_a_real_run():
    final = G.run_sequential(G.example_inputs("abstain"))
    assert decision_banner(final)[0] == "error"
    assert all("⚠" in finding_label(f) for f in final["fused"]["findings"] if f["conflict"])
    assert {r["pmid"] for r in cited_refs(final)} == set(final["citations"])
    assert trace_line(final["trace"][0]).startswith("🧭 **router**")


def test_report_for_display_shrinks_only_headings():
    final = G.run_sequential(G.example_inputs("conflict"))
    shown = report_for_display(final["report_md"])
    assert "#### Résumé" in shown and "\n## " not in shown and DISCLAIMER in shown


def test_evidence_gives_french_names_from_the_notes():
    ev = E.evidence(G.run_sequential(G.example_inputs("conflict")))
    assert ev["names_fr"]["Effusion"] == "Épanchement pleural"
    assert set(ev["names_fr"]) == {f["label"] for f in ev["findings"]}
