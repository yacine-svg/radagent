"""B1 FAKE agents: return the matching parts of examples/*.json, adapted to the current case.

They let the whole graph run before A's models and B's MCP servers exist.
B3 replaces them with the real agents (same names, same outputs, same contract).
Which example is used: case_id "demo-agreement" / "demo-conflict" / "demo-abstain"
(anything else -> the agreement example).
"""
from __future__ import annotations

import copy
import json
import time

from contracts import HISTORY_SEPARATOR, heatmap_path
from agents.common import ROOT, empty_text_result, event, safe_node

EXAMPLES = {"demo-agreement": "case_agreement",
            "demo-conflict": "case_conflict_warning",
            "demo-abstain": "case_abstain"}


def _example(state: dict) -> dict:
    name = EXAMPLES.get(state.get("case_id", ""), "case_agreement")
    return json.loads((ROOT / "examples" / f"{name}.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ image agent
def _image_failed(state, msg):
    return {"image_result": {"findings": [], "model": "", "heatmaps": {}, "error": msg}}


@safe_node("image_agent", _image_failed)
def image_agent(state: dict) -> dict:
    t0 = time.perf_counter()
    res = copy.deepcopy(_example(state)["image_result"])
    res["heatmaps"] = {label: heatmap_path(state["case_id"], label) for label in res["heatmaps"]}
    above = [f["label"] for f in res["findings"] if f["above_threshold"]]
    msg = f"FAKE — {len(res['findings'])} findings scored, above threshold: {', '.join(above) or 'none'}"
    return {"image_result": res, "trace": [event("image_agent", msg, t0, "vision-server.classify_image")]}


# ------------------------------------------------------------------ text agent
def _text_failed(state, msg):
    return {"text_result": empty_text_result(error=msg), "patient": None}


@safe_node("text_agent", _text_failed)
def text_agent(state: dict) -> dict:
    t0 = time.perf_counter()
    ex = _example(state)
    pid = state.get("patient_id")

    # patient lookup (patient-server in B3)
    patient = None
    if pid is not None:
        rec = ex.get("patient")
        patient = copy.deepcopy(rec) if rec and rec.get("patient_id") == pid else {"patient_id": pid, "found": False}

    if not state["flags"]["has_text"]:
        return {"text_result": empty_text_result(), "patient": patient,
                "trace": [event("text_agent", "FAKE — no clinical text", t0)]}

    res = copy.deepcopy(ex["text_result"])
    history_ok = bool(patient and patient.get("found") and patient.get("history"))
    if res["history_used"] and not history_ok:          # keep the contract true without history
        cut = res["analyzed_text"].split(HISTORY_SEPARATOR)[0]
        res["entities"] = [e for e in res["entities"] if e["span"][1] <= len(cut)]
        res["analyzed_text"], res["history_used"] = cut, False
    tool = "patient-server.get_patient_history" if pid else None
    msg = (f"FAKE — {len(res['entities'])} entities, "
           f"{sum(e['negated'] for e in res['entities'])} negated"
           + (", patient history added" if res["history_used"] else ""))
    return {"text_result": res, "patient": patient, "trace": [event("text_agent", msg, t0, tool)]}


# ------------------------------------------------------------------ knowledge agent
def _knowledge_failed(state, msg):
    return {"knowledge": {"by_class": {}, "error": msg}}


@safe_node("knowledge_agent", _knowledge_failed)
def knowledge_agent(state: dict) -> dict:
    t0 = time.perf_counter()
    res = copy.deepcopy(_example(state)["knowledge"])
    n = sum(len(k["refs"]) for k in res["by_class"].values())
    msg = f"FAKE — ICD-10 + {n} references + guideline notes for {len(res['by_class'])} classes"
    return {"knowledge": res, "trace": [event("knowledge_agent", msg, t0, "knowledge-server.search_pubmed")]}
