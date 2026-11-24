"""Router node: validates the inputs, sets case_id and flags (contracts.RouterFlags)."""
from __future__ import annotations

import pathlib
import time
import uuid

from contracts import CONTRACT_VERSION
from agents.common import ROOT, event

IMAGE_TYPES = {".png", ".jpg", ".jpeg"}


def router(state: dict) -> dict:
    t0 = time.perf_counter()
    text = state.get("text") or ""
    patient_id = state.get("patient_id") or None
    case_id = state.get("case_id") or f"case-{uuid.uuid4().hex[:8]}"

    errors = []
    path = state.get("image_path") or ""
    if not path:
        errors.append("no image given")
    elif pathlib.Path(path).suffix.lower() not in IMAGE_TYPES:
        errors.append(f"unsupported image type: {pathlib.Path(path).suffix or '(none)'} (PNG or JPG)")
    elif not (pathlib.Path(path).exists() or (ROOT / path).exists()):
        errors.append(f"image file not found: {path}")

    flags = {"has_text": bool(text.strip()), "has_patient": patient_id is not None, "errors": errors}
    parts = ["image"] + (["clinical text"] if flags["has_text"] else []) + (["patient id"] if patient_id else [])
    msg = "Inputs: " + " + ".join(parts) + (f" — warnings: {'; '.join(errors)}" if errors else "")
    return {"contract_version": CONTRACT_VERSION, "case_id": case_id, "image_path": path,
            "text": text, "patient_id": patient_id, "flags": flags,
            "trace": [event("router", msg, t0)]}
