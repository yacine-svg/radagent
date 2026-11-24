"""RadAgent data contract — executable version of contracts.md.

Every JSON exchanged between components is defined here as a pydantic model.
Validate any output with e.g.  ImageResult.model_validate(obj)  — it raises with a
clear message if a key, type, class name, range or business rule is wrong.

Rules:
- Change this file only when A and B both agree, and bump CONTRACT_VERSION.
- Class names come from ml/labels.py (never hard-code them elsewhere).
- The fusion formula and the conflict rule live HERE, so A's evaluation and B's agents
  use exactly the same maths (agents/fusion.py re-exports them).
"""
from __future__ import annotations

import math
import re
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml.labels import CLASSES

CONTRACT_VERSION = "1.0"

# ---------------------------------------------------------------- constants
CONFLICT_HIGH = 0.70          # image "clearly yes"
CONFLICT_LOW = 0.30           # image "clearly no"
UNCERTAIN_BAND = (0.35, 0.65) # fused probability considered "unsure" (inclusive)
PROB_DECIMALS = 4             # every probability is rounded to 4 decimals
TOLERANCE = 1e-3              # allowed rounding difference in cross-checks
DISCLAIMER = "Aide à l'analyse uniquement — ne constitue pas un diagnostic médical."
PMID_PATTERN = re.compile(r"PMID:\s*(\d{1,9})")
REPORT_SECTIONS = ["## Résumé", "## Observations image", "## Éléments cliniques",
                   "## Points de vigilance", "## Références", "## Avertissement"]
HISTORY_SEPARATOR = " | Patient history: "   # text agent: analyzed text = text + SEP + "; ".join(history)

Prob = Annotated[float, Field(ge=0.0, le=1.0)]
Signal = Literal[-1, 0, 1]
Status = Literal["positive", "negated", "absent"]
STATUS_TO_SIGNAL = {"positive": 1, "negated": -1, "absent": 0}
NodeName = Literal["router", "image_agent", "text_agent", "knowledge_agent",
                   "fusion", "critic", "abstain", "explainer"]


# ---------------------------------------------------------------- shared maths
def fuse(p_image: float, s_text: int, beta: float) -> float:
    """Log-odds fusion: the text signal (-1/0/+1) shifts the image log-odds by beta."""
    p = min(max(float(p_image), 1e-4), 1 - 1e-4)
    z = math.log(p / (1 - p)) + beta * int(s_text)
    return 1.0 / (1.0 + math.exp(-z))


def detect_conflict(p_image: float, s_text: int) -> Optional[str]:
    """Conflict rule. Returns the conflict type or None."""
    if p_image >= CONFLICT_HIGH and s_text == -1:
        return "image_pos_text_neg"
    if p_image <= CONFLICT_LOW and s_text == 1:
        return "text_pos_image_neg"
    return None


def in_uncertain_band(p: float) -> bool:
    return UNCERTAIN_BAND[0] <= p <= UNCERTAIN_BAND[1]


def label_slug(label: str) -> str:
    """'Lung Opacity' -> 'Lung_Opacity' (used in file names)."""
    return label.replace(" ", "_")


def heatmap_path(case_id: str, label: str) -> str:
    """Path convention for Grad-CAM overlays (relative to repo root, forward slashes)."""
    return f"outputs/{case_id}/cam_{label_slug(label)}.png"


def _check_classes(labels: list[str], where: str) -> None:
    if labels != CLASSES:
        raise ValueError(f"{where}: labels must be exactly {CLASSES} in this order, got {labels}")


class Strict(BaseModel):
    """Base: unknown keys are an error (catches typos like 'probability' vs 'prob')."""
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- 1. input & router
class CaseInput(Strict):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    image_path: str
    text: str = ""
    patient_id: Optional[str] = None


class RouterFlags(Strict):
    has_text: bool
    has_patient: bool
    errors: list[str] = []


# ---------------------------------------------------------------- 2. vision (Person A)
class ImageFinding(Strict):
    label: str
    prob: Prob
    above_threshold: bool


class VisionPrediction(Strict):
    """Output of ml.vision.predict(path) and of MCP tool vision-server.classify_image."""
    findings: list[ImageFinding]
    model: str

    @model_validator(mode="after")
    def _classes(self):
        _check_classes([f.label for f in self.findings], "VisionPrediction.findings")
        return self


class ImageResult(Strict):
    """state['image_result'] — written by the Image Agent (Person B)."""
    findings: list[ImageFinding] = []
    model: str = ""
    heatmaps: dict[str, str] = {}          # label -> heatmap_path(case_id, label)
    error: Optional[str] = None

    @model_validator(mode="after")
    def _rules(self):
        if self.error is not None:
            return self                    # on error, findings may be empty
        _check_classes([f.label for f in self.findings], "ImageResult.findings")
        above = [f.label for f in self.findings if f.above_threshold]
        expected = above or [max(self.findings, key=lambda f: f.prob).label]
        if sorted(self.heatmaps) != sorted(expected):
            raise ValueError(f"ImageResult.heatmaps: expected heatmaps for {expected} "
                             f"(above-threshold classes, or top-1 if none), got {sorted(self.heatmaps)}")
        return self


# ---------------------------------------------------------------- 3. text (Person A)
class Entity(Strict):
    text: str = Field(min_length=1)
    type: Literal["symptom", "history", "finding"]
    negated: bool
    span: tuple[int, int]                  # [start, end) in analyzed_text


class TextResult(Strict):
    """Output of ml.text_nlp.analyze_text(text) and state['text_result']."""
    analyzed_text: str                     # exact string the spans refer to
    entities: list[Entity] = []
    suspected: dict[str, Status]
    history_used: bool = False             # set True by the Text Agent when it appended patient history
    error: Optional[str] = None

    @model_validator(mode="after")
    def _rules(self):
        if set(self.suspected) != set(CLASSES):
            raise ValueError(f"TextResult.suspected: keys must be {CLASSES}, got {list(self.suspected)}")
        for e in self.entities:
            s, t = e.span
            if not (0 <= s < t <= len(self.analyzed_text)):
                raise ValueError(f"Entity '{e.text}': span {e.span} outside analyzed_text")
            if self.analyzed_text[s:t] != e.text:
                raise ValueError(f"Entity '{e.text}': analyzed_text[{s}:{t}] is "
                                 f"'{self.analyzed_text[s:t]}' (must match exactly)")
        return self

    def signal(self) -> dict[str, int]:
        return {c: STATUS_TO_SIGNAL[self.suspected[c]] for c in CLASSES}


# ---------------------------------------------------------------- 4. patient-server (Person B)
class PatientRecord(Strict):
    """Output of MCP tool patient-server.get_patient_history. SYNTHETIC data only."""
    patient_id: str
    found: bool
    age: Optional[int] = Field(default=None, ge=0, le=120)
    sex: Optional[Literal["M", "F"]] = None
    smoker: Optional[bool] = None
    history: list[str] = []
    prior_studies: list[str] = []


# ---------------------------------------------------------------- 5. knowledge-server (Person B)
class ICD10(Strict):
    """Output of MCP tool knowledge-server.icd10_lookup."""
    condition: str
    code: Optional[str] = Field(default=None, pattern=r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$")
    label: Optional[str] = None


class Reference(Strict):
    """One item of MCP tool knowledge-server.search_pubmed (which returns a list)."""
    pmid: str = Field(pattern=r"^\d{1,9}$")
    title: str
    journal: str = ""
    year: str = Field(default="", pattern=r"^(\d{4})?$")


class GuidelineSnippet(Strict):
    """One item of MCP tool knowledge-server.guideline_snippets (which returns a list)."""
    text: str
    source: str                            # URL of the page the note was written from


class ClassKnowledge(Strict):
    icd10: ICD10
    refs: list[Reference] = []
    guideline: Optional[GuidelineSnippet] = None


class KnowledgeResult(Strict):
    """state['knowledge'] — evidence prefetched for EVERY class by the Knowledge Agent."""
    by_class: dict[str, ClassKnowledge] = {}
    error: Optional[str] = None

    @model_validator(mode="after")
    def _rules(self):
        if self.error is None and list(self.by_class) != CLASSES:
            raise ValueError(f"KnowledgeResult.by_class: keys must be {CLASSES} in order")
        return self

    def all_pmids(self) -> set[str]:
        return {r.pmid for k in self.by_class.values() for r in k.refs}


# ---------------------------------------------------------------- 6. fusion (Person B)
class Conflict(Strict):
    type: Literal["image_pos_text_neg", "text_pos_image_neg"]
    message: str


class FusedFinding(Strict):
    label: str
    p_image: Prob
    s_text: Signal
    p_fused: Prob
    threshold: Prob                        # thresholds.json["fused"][label]
    above_threshold: bool                  # p_fused >= threshold
    conflict: Optional[Conflict] = None

    @model_validator(mode="after")
    def _rules(self):
        if self.above_threshold != (self.p_fused >= self.threshold):
            raise ValueError(f"{self.label}: above_threshold must equal p_fused >= threshold")
        expected = detect_conflict(self.p_image, self.s_text)
        got = self.conflict.type if self.conflict else None
        if expected != got:
            raise ValueError(f"{self.label}: conflict should be {expected}, got {got}")
        return self


class FusedResult(Strict):
    """state['fused']"""
    findings: list[FusedFinding]
    beta: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _rules(self):
        _check_classes([f.label for f in self.findings], "FusedResult.findings")
        for f in self.findings:
            if abs(fuse(f.p_image, f.s_text, self.beta) - f.p_fused) > TOLERANCE:
                raise ValueError(f"{f.label}: p_fused {f.p_fused} != fuse(p_image, s_text, beta) "
                                 f"= {fuse(f.p_image, f.s_text, self.beta):.4f}")
        return self


# ---------------------------------------------------------------- 7. critic (Person B)
ReasonCode = Literal[
    "uncertain_conflict",   # abstain: conflict AND p_fused in UNCERTAIN_BAND
    "all_uncertain",        # abstain: every class has p_fused in UNCERTAIN_BAND
    "image_failed",         # abstain: no image result -> nothing to say
    "conflict",             # warning: conflict but p_fused outside the band
    "text_failed",          # warning: text analysis failed, fusion used s_text = 0
    "knowledge_failed",     # warning: no references available
    "patient_not_found",    # info
    "no_text",              # info: no clinical text given, fusion used s_text = 0
    "entity_dropped",       # info: an extracted entity was not found verbatim
]


class Reason(Strict):
    code: ReasonCode
    severity: Literal["info", "warning", "abstain"]
    label: Optional[str] = None            # the class concerned, if any
    message: str

    @model_validator(mode="after")
    def _severity_matches_code(self):
        abstain_codes = {"uncertain_conflict", "all_uncertain", "image_failed"}
        warning_codes = {"conflict", "text_failed", "knowledge_failed"}
        expected = ("abstain" if self.code in abstain_codes
                    else "warning" if self.code in warning_codes else "info")
        if self.severity != expected:
            raise ValueError(f"reason '{self.code}' must have severity '{expected}'")
        if self.label is not None and self.label not in CLASSES:
            raise ValueError(f"reason label '{self.label}' is not a class")
        return self


# ---------------------------------------------------------------- 8. trace
class TraceEvent(Strict):
    node: NodeName
    message: str                           # one human-readable line for the live UI
    duration_ms: int = Field(ge=0)
    tool: Optional[str] = None             # e.g. "vision-server.classify_image"


# ---------------------------------------------------------------- 9. final state
class FinalState(Strict):
    """Everything the graph returns. The UI renders only from this."""
    contract_version: Literal["1.0"] = CONTRACT_VERSION
    case_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    image_path: str
    text: str = ""
    patient_id: Optional[str] = None
    flags: RouterFlags
    image_result: ImageResult
    text_result: TextResult
    patient: Optional[PatientRecord] = None
    knowledge: KnowledgeResult
    fused: Optional[FusedResult] = None    # None only when image_failed
    decision: Literal["ok", "abstain"]
    reasons: list[Reason] = []
    report_md: str
    citations: list[str] = []              # PMIDs cited in report_md
    trace: list[TraceEvent]

    @model_validator(mode="after")
    def _cross_checks(self):
        codes = {(r.code, r.label) for r in self.reasons}

        # router
        if self.flags.has_text != bool(self.text.strip()):
            raise ValueError("flags.has_text must be bool(text.strip())")
        if self.flags.has_patient != (self.patient_id is not None):
            raise ValueError("flags.has_patient must be (patient_id is not None)")

        # heatmap paths follow the convention
        for label, path in self.image_result.heatmaps.items():
            if path != heatmap_path(self.case_id, label):
                raise ValueError(f"heatmap for {label} must be at {heatmap_path(self.case_id, label)}")

        # patient / history
        if self.patient_id is None and self.patient is not None:
            raise ValueError("patient must be None when patient_id is None")
        if self.text_result.history_used and not (self.patient and self.patient.found and self.patient.history):
            raise ValueError("history_used=True requires a found patient with a non-empty history")
        if self.patient is not None and not self.patient.found and ("patient_not_found", None) not in codes:
            raise ValueError("patient not found -> add reason 'patient_not_found'")

        # image failure
        if self.image_result.error is not None:
            if ("image_failed", None) not in codes:
                raise ValueError("image_result.error set -> add reason 'image_failed'")
            if self.fused is not None:
                raise ValueError("fused must be None when the image failed")
        else:
            if self.fused is None:
                raise ValueError("fused is required when the image succeeded")
            img = {f.label: f.prob for f in self.image_result.findings}
            sig = ({c: 0 for c in CLASSES} if self.text_result.error is not None
                   else self.text_result.signal())
            for f in self.fused.findings:
                if abs(f.p_image - img[f.label]) > TOLERANCE:
                    raise ValueError(f"fused.{f.label}.p_image must equal image_result prob")
                if f.s_text != sig[f.label]:
                    raise ValueError(f"fused.{f.label}.s_text must equal the text signal ({sig[f.label]})")
                if f.conflict:
                    code = "uncertain_conflict" if in_uncertain_band(f.p_fused) else "conflict"
                    if (code, f.label) not in codes:
                        raise ValueError(f"{f.label}: conflict -> add reason '{code}' with label '{f.label}'")
            if all(in_uncertain_band(f.p_fused) for f in self.fused.findings) \
                    and ("all_uncertain", None) not in codes:
                raise ValueError("every class is uncertain -> add reason 'all_uncertain'")

        # other failures / missing inputs
        if self.text_result.error is not None and ("text_failed", None) not in codes:
            raise ValueError("text_result.error set -> add reason 'text_failed'")
        if self.knowledge.error is not None and ("knowledge_failed", None) not in codes:
            raise ValueError("knowledge.error set -> add reason 'knowledge_failed'")
        if not self.flags.has_text and ("no_text", None) not in codes:
            raise ValueError("no text -> add reason 'no_text'")

        # decision follows the reasons
        should_abstain = any(r.severity == "abstain" for r in self.reasons)
        if (self.decision == "abstain") != should_abstain:
            raise ValueError("decision must be 'abstain' exactly when a reason has severity 'abstain'")

        # report structure + grounding
        pos = [self.report_md.find(h) for h in REPORT_SECTIONS]
        if -1 in pos or pos != sorted(pos):
            raise ValueError(f"report_md must contain these headings in this order: {REPORT_SECTIONS}")
        if DISCLAIMER not in self.report_md:
            raise ValueError("report_md must contain the DISCLAIMER sentence verbatim")
        cited = set(PMID_PATTERN.findall(self.report_md))
        if cited != set(self.citations):
            raise ValueError(f"citations {sorted(self.citations)} must equal PMIDs in report_md {sorted(cited)}")
        unknown = cited - self.knowledge.all_pmids()
        if unknown:
            raise ValueError(f"report cites PMIDs not returned by the knowledge agent: {sorted(unknown)}")

        # trace shape
        nodes = [t.node for t in self.trace]
        if not nodes or nodes[0] != "router" or nodes[-1] != "explainer":
            raise ValueError("trace must start with 'router' and end with 'explainer'")
        if ("abstain" in nodes) != (self.decision == "abstain"):
            raise ValueError("trace contains 'abstain' exactly when decision == 'abstain'")
        return self
