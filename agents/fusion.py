"""Fusion maths (re-exported from contracts.py) + the fusion node.

A imports the maths from here (`from agents.fusion import fuse`). Never redefine fuse(),
detect_conflict() or the constants in this file: they live in contracts.py (AGENTS.md 3.4).
"""
from __future__ import annotations

import time

from contracts import (CONFLICT_HIGH, CONFLICT_LOW, STATUS_TO_SIGNAL, UNCERTAIN_BAND,  # noqa: F401
                       detect_conflict, fuse, in_uncertain_band)
from ml.labels import CLASSES
from agents.common import event, load_beta, load_fused_thresholds

CONFLICT_MESSAGES = {
    "image_pos_text_neg": "Image suggests {label} (p={p:.2f}) but the clinical text says it is absent.",
    "text_pos_image_neg": "Clinical text suggests {label} but the image does not (p={p:.2f}).",
}


def text_signal(text_result: dict) -> dict[str, int]:
    """-1/0/+1 per class; 0 everywhere if the text analysis failed (contract §8)."""
    if text_result.get("error") is not None:
        return {c: 0 for c in CLASSES}
    return {c: STATUS_TO_SIGNAL[text_result["suspected"][c]] for c in CLASSES}


def fusion(state: dict) -> dict:
    """state['fused'] = contracts.FusedResult (None when the image failed)."""
    t0 = time.perf_counter()
    image = state["image_result"]
    if image.get("error") is not None:
        return {"fused": None, "trace": [event("fusion", "Skipped — no image result", t0)]}

    signal = text_signal(state["text_result"])
    beta, thresholds = load_beta(), load_fused_thresholds()
    findings = []
    for f in image["findings"]:
        label, p, s = f["label"], f["prob"], signal[f["label"]]
        p_fused = round(fuse(p, s, beta), 4)
        kind = detect_conflict(p, s)
        findings.append({
            "label": label, "p_image": p, "s_text": s, "p_fused": p_fused,
            "threshold": thresholds[label], "above_threshold": p_fused >= thresholds[label],
            "conflict": {"type": kind, "message": CONFLICT_MESSAGES[kind].format(label=label, p=p)} if kind else None,
        })
    n_conf = sum(f["conflict"] is not None for f in findings)
    msg = f"Image and text combined (beta={beta}) — {n_conf} conflict(s)"
    return {"fused": {"findings": findings, "beta": beta}, "trace": [event("fusion", msg, t0)]}
