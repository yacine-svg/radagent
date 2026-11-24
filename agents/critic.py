"""Critic node: deterministic rules from contracts.md §9 -> decision + reasons."""
from __future__ import annotations

import time

from contracts import in_uncertain_band
from agents.common import event


def _reason(code, severity, message, label=None):
    return {"code": code, "severity": severity, "label": label, "message": message}


def critic(state: dict) -> dict:
    t0 = time.perf_counter()
    reasons = []

    image = state["image_result"]
    if image.get("error") is not None:
        reasons.append(_reason("image_failed", "abstain", f"Image analysis failed: {image['error']}"))

    fused = state.get("fused")
    if fused:
        for f in fused["findings"]:
            if f["conflict"]:
                if in_uncertain_band(f["p_fused"]):
                    reasons.append(_reason("uncertain_conflict", "abstain",
                                           f"{f['conflict']['message']} Combined probability {f['p_fused']:.2f} is uncertain.",
                                           f["label"]))
                else:
                    reasons.append(_reason("conflict", "warning",
                                           f"{f['conflict']['message']} Combined probability stays at {f['p_fused']:.2f}.",
                                           f["label"]))
        if all(in_uncertain_band(f["p_fused"]) for f in fused["findings"]):
            reasons.append(_reason("all_uncertain", "abstain",
                                   "Every finding is in the uncertain band: no reliable conclusion."))

    if state["text_result"].get("error") is not None:
        reasons.append(_reason("text_failed", "warning",
                               f"Text analysis failed ({state['text_result']['error']}); image evidence only."))
    if state["knowledge"].get("error") is not None:
        reasons.append(_reason("knowledge_failed", "warning", "No references available for this case."))
    patient = state.get("patient")
    if patient is not None and not patient.get("found"):
        reasons.append(_reason("patient_not_found", "info", f"Patient {patient['patient_id']} not found; no history used."))
    if not state["flags"]["has_text"]:
        reasons.append(_reason("no_text", "info", "No clinical text given; image evidence only."))

    decision = "abstain" if any(r["severity"] == "abstain" for r in reasons) else "ok"
    counts = {s: sum(r["severity"] == s for r in reasons) for s in ("abstain", "warning", "info")}
    msg = f"Decision: {decision} — {counts['abstain']} blocking, {counts['warning']} warning(s), {counts['info']} note(s)"
    return {"decision": decision, "reasons": reasons, "trace": [event("critic", msg, t0)]}
