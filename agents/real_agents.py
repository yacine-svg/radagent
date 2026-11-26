"""B3 real agents: they call our MCP servers (agents/mcp_client.py) and Person A's text module.

image_agent     -> vision-server.classify_image + gradcam            (needs ml/vision.py, A2)
text_agent      -> patient-server.get_patient_history + ml.text_nlp   (placeholder analysis until A3)
knowledge_agent -> knowledge-server: icd10_lookup, search_pubmed, guideline_snippets for every class
"""
from __future__ import annotations

import copy
import importlib.util
import json
import time

from mcp import Client

from contracts import HISTORY_SEPARATOR
from ml.labels import CLASSES
from agents.common import ROOT, empty_text_result, event, root_cause, safe_node
from agents.fake_agents import _image_failed, _knowledge_failed, _text_failed
from agents.mcp_client import target, tool_output
from servers.knowledge import pubmed_query

REFS_PER_CLASS = 2


# ------------------------------------------------------------------ image agent
@safe_node("image_agent", _image_failed)
async def image_agent(state: dict) -> dict:
    t0 = time.perf_counter()
    async with Client(target("vision")) as c:
        pred = tool_output(await c.call_tool("classify_image", {"image_path": state["image_path"]}))
        above = [f["label"] for f in pred["findings"] if f["above_threshold"]]
        cam_labels = above or [max(pred["findings"], key=lambda f: f["prob"])["label"]]   # contract §4
        heatmaps = {}
        for label in cam_labels:
            heatmaps[label] = tool_output(await c.call_tool(
                "gradcam", {"image_path": state["image_path"], "label": label, "case_id": state["case_id"]}))
    msg = f"{len(pred['findings'])} findings scored, above threshold: {', '.join(above) or 'none'}; {len(heatmaps)} heatmap(s)"
    return {"image_result": {**pred, "heatmaps": heatmaps, "error": None},
            "trace": [event("image_agent", msg, t0, "vision-server.classify_image")]}


# ------------------------------------------------------------------ text agent
def _placeholder_analysis(text: str) -> dict:
    """Used only until A merges ml/text_nlp.py: reuses an example's result when the text is
    one of the demo texts, otherwise returns 'nothing found' (clearly marked in the trace)."""
    norm = " ".join(text.split())
    for path in sorted((ROOT / "examples").glob("case_*.json")):
        ex = json.loads(path.read_text(encoding="utf-8"))["text_result"]
        if ex["analyzed_text"] == norm:
            return {**copy.deepcopy(ex), "history_used": False, "error": None}
    return {**empty_text_result(), "analyzed_text": norm}


def _analyze(text: str) -> tuple[dict, str]:
    if importlib.util.find_spec("ml.text_nlp") is None:
        return _placeholder_analysis(text), "placeholder analysis (ml/text_nlp.py not merged yet)"
    from ml.text_nlp import analyze_text
    return analyze_text(text), "LLM analysis"


@safe_node("text_agent", _text_failed)
async def text_agent(state: dict) -> dict:
    t0 = time.perf_counter()
    pid, text = state.get("patient_id"), state.get("text") or ""
    notes, tool = [], None

    patient = None
    if pid is not None:
        tool = "patient-server.get_patient_history"
        try:
            async with Client(target("patient")) as c:
                patient = tool_output(await c.call_tool("get_patient_history", {"patient_id": pid}))
            notes.append(f"patient {pid} {'found' if patient['found'] else 'not found'}")
        except Exception as e:                           # noqa: BLE001 — keep going without history
            notes.append(f"patient lookup failed ({type(root_cause(e)).__name__})")

    if not state["flags"]["has_text"]:
        return {"text_result": empty_text_result(), "patient": patient,
                "trace": [event("text_agent", "; ".join(["no clinical text", *notes]), t0, tool)]}

    history = patient["history"] if patient and patient.get("found") else []
    full_text = text + HISTORY_SEPARATOR + "; ".join(history) if history else text
    try:
        result, how = _analyze(full_text)
        result["history_used"] = bool(history) and HISTORY_SEPARATOR.strip() in result["analyzed_text"]
        msg = (f"{how}: {len(result['entities'])} entities, {sum(e['negated'] for e in result['entities'])} negated"
               + (", patient history added" if result["history_used"] else ""))
    except Exception as e:                               # noqa: BLE001 — text failure keeps the patient
        result = empty_text_result(error=f"{type(e).__name__}: {e}")
        msg = f"text analysis FAILED — {result['error']}"
    return {"text_result": result, "patient": patient,
            "trace": [event("text_agent", "; ".join([msg, *notes]), t0, tool)]}


# ------------------------------------------------------------------ knowledge agent
@safe_node("knowledge_agent", _knowledge_failed)
async def knowledge_agent(state: dict) -> dict:
    """Prefetches evidence for EVERY class (contract §7), so it runs in parallel with the others."""
    t0 = time.perf_counter()
    by_class = {}
    async with Client(target("knowledge")) as c:
        for label in CLASSES:
            icd = tool_output(await c.call_tool("icd10_lookup", {"condition": label}))
            refs = tool_output(await c.call_tool("search_pubmed", {"query": pubmed_query(label),
                                                                   "max_results": REFS_PER_CLASS}))
            notes = tool_output(await c.call_tool("guideline_snippets", {"condition": label, "k": 1}))
            by_class[label] = {"icd10": icd, "refs": refs, "guideline": notes[0] if notes else None}
    n_refs = sum(len(k["refs"]) for k in by_class.values())
    msg = f"ICD-10, {n_refs} reference(s) and guideline notes for {len(by_class)} classes"
    if n_refs == 0:
        msg += " — no PubMed results (offline and cache empty? run python -m servers.warm_cache)"
    return {"knowledge": {"by_class": by_class, "error": None},
            "trace": [event("knowledge_agent", msg, t0, "knowledge-server")]}
