"""Contract tests. Run from the repo root:  python -m pytest tests/test_contracts.py -q

- The 3 example cases must validate.
- Typical mistakes must be REJECTED (so we know the contract really checks them).
- Live checks of A's real functions run only with RUN_ML_TESTS=1 (they load the model / call Groq).
"""
import copy, json, os, pathlib
import pytest
from pydantic import ValidationError

from contracts import (CLASSES, DISCLAIMER, REPORT_SECTIONS, FinalState, TextResult,
                       VisionPrediction, fuse, detect_conflict)

EX = pathlib.Path(__file__).resolve().parent.parent / "examples"
CASES = sorted(EX.glob("case_*.json"))


def load(name):
    return json.loads((EX / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_examples_are_valid(path):
    FinalState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def test_fusion_maths():
    assert fuse(0.8, 0, 1.0) == pytest.approx(0.8, abs=1e-6)          # no text -> unchanged
    assert fuse(0.8, 1, 1.0) > 0.8 > fuse(0.8, -1, 1.0)                # text moves it both ways
    assert fuse(0.8, -1, 0.0) == pytest.approx(0.8, abs=1e-6)          # beta 0 ignores text
    assert detect_conflict(0.9, -1) == "image_pos_text_neg"
    assert detect_conflict(0.1, 1) == "text_pos_image_neg"
    assert detect_conflict(0.5, -1) is None


def test_image_failure_state_is_valid():
    """Failures are part of the contract: the graph must still return a valid state."""
    o = load("case_conflict_warning")
    o["image_result"] = {"findings": [], "model": "", "heatmaps": {}, "error": "cannot read image"}
    o["fused"] = None
    o["text"], o["flags"]["has_text"] = "", False
    o["text_result"] = {"analyzed_text": "", "entities": [], "history_used": False, "error": None,
                        "suspected": {c: "absent" for c in CLASSES}}
    o["decision"] = "abstain"
    o["reasons"] = [{"code": "image_failed", "severity": "abstain", "label": None, "message": "Unreadable image."},
                    {"code": "no_text", "severity": "info", "label": None, "message": "No clinical text."}]
    o["report_md"] = "\n\n".join(REPORT_SECTIONS[:-1]) + "\n\n" + REPORT_SECTIONS[-1] + "\n" + DISCLAIMER
    o["citations"] = []
    o["trace"].insert(-1, {"node": "abstain", "message": "Referred for review", "duration_ms": 1, "tool": None})
    FinalState.model_validate(o)


def test_text_failure_forces_zero_signal():
    o = load("case_conflict_warning")
    o["text_result"]["error"] = "groq timeout"
    with pytest.raises(ValidationError, match="s_text must equal the text signal"):
        FinalState.model_validate(o)


# ---- each mutation below must be rejected ----------------------------------
def _mut(case, fn):
    obj = copy.deepcopy(load(case)); fn(obj); return obj


BROKEN = {
    "unknown key": ("case_agreement", lambda o: o["image_result"]["findings"][0].update(probability=0.8)),
    "prob > 1": ("case_agreement", lambda o: o["image_result"]["findings"][0].update(prob=1.3)),
    "class order": ("case_agreement", lambda o: o["image_result"]["findings"].reverse()),
    "wrong class name": ("case_agreement", lambda o: o["text_result"]["suspected"].update(
        {"Pleural Effusion": o["text_result"]["suspected"].pop("Effusion")})),
    "span mismatch": ("case_agreement", lambda o: o["text_result"]["entities"][0].update(span=[0, 5])),
    "missing heatmap": ("case_agreement", lambda o: o["image_result"].update(heatmaps={})),
    "bad heatmap path": ("case_agreement", lambda o: o["image_result"]["heatmaps"].update(
        Cardiomegaly="outputs/cam.png")),
    "p_fused not from formula": ("case_agreement", lambda o: o["fused"]["findings"][0].update(p_fused=0.99)),
    "missing conflict": ("case_conflict_warning", lambda o: o["fused"]["findings"][1].update(conflict=None)),
    "conflict not reported": ("case_conflict_warning", lambda o: o.update(reasons=[])),
    "decision vs reasons": ("case_abstain", lambda o: o.update(decision="ok")),
    "wrong severity": ("case_abstain", lambda o: o["reasons"][0].update(severity="warning")),
    "uncited PMID": ("case_agreement", lambda o: o.update(report_md=o["report_md"] + "\nPMID: 12345678")),
    "citation not in report": ("case_agreement", lambda o: o.update(citations=["90000001", "90000002"])),
    "no disclaimer": ("case_agreement", lambda o: o.update(report_md=o["report_md"].replace(DISCLAIMER, ""))),
    "missing report section": ("case_agreement", lambda o: o.update(
        report_md=o["report_md"].replace("## Points de vigilance", "## Notes"))),
    "trace order": ("case_agreement", lambda o: o["trace"].reverse()),
    "history flag": ("case_conflict_warning", lambda o: o["text_result"].update(history_used=True)),
}


@pytest.mark.parametrize("name", list(BROKEN))
def test_contract_rejects(name):
    case, fn = BROKEN[name]
    with pytest.raises(ValidationError):
        FinalState.model_validate(_mut(case, fn))


# ---- live checks of Person A's functions (opt-in) --------------------------
LIVE = pytest.mark.skipif(os.environ.get("RUN_ML_TESTS") != "1", reason="set RUN_ML_TESTS=1 to run")


@LIVE
def test_vision_predict_matches_contract():
    from ml import vision
    img = next(pathlib.Path("data/demo").glob("*.png"))
    VisionPrediction.model_validate(vision.predict(str(img)))


@LIVE
def test_analyze_text_matches_contract():
    from ml.text_nlp import analyze_text
    r = TextResult.model_validate(analyze_text("XXXX-year-old woman, history of CHF, no fever"))
    assert set(r.suspected) == set(CLASSES)
    assert TextResult.model_validate(analyze_text("")).entities == []
