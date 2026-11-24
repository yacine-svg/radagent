"""patient-server logic: get_patient_history(patient_id) -> contracts.PatientRecord. Never raises."""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "patients.db"
log = logging.getLogger("patient-server")


def _ensure_db() -> None:
    if not DB.exists():
        from data.make_patients import build
        build(DB)


def get_patient_history(patient_id: str) -> dict:
    pid = str(patient_id).strip()
    try:
        _ensure_db()
        with sqlite3.connect(DB) as con:
            row = con.execute("SELECT patient_id, age, sex, smoker, history, prior_studies "
                              "FROM patients WHERE patient_id = ?", (pid,)).fetchone()
    except Exception as e:                                # noqa: BLE001 — AGENTS.md 6.1
        log.warning("patient lookup failed for %r: %s", pid, e)
        row = None
    if row is None:
        return {"patient_id": pid, "found": False}
    _, age, sex, smoker, history, prior = row
    return {"patient_id": pid, "found": True, "age": age, "sex": sex, "smoker": bool(smoker),
            "history": json.loads(history), "prior_studies": json.loads(prior)}
