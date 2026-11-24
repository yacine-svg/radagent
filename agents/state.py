"""LangGraph state — mirrors contracts.FinalState (dicts inside, validated with contracts.py).

Who writes what (each key has ONE writer, except `trace`, which every node appends to):
  router          -> case_id (if missing), flags, trace
  image_agent     -> image_result, trace
  text_agent      -> text_result, patient, trace
  knowledge_agent -> knowledge, trace
  fusion          -> fused, trace
  critic          -> decision, reasons, trace
  abstain         -> trace   (only runs when decision == "abstain")
  explainer       -> report_md, citations, trace
"""
import operator
from typing import Annotated, Optional, TypedDict


class CaseState(TypedDict, total=False):
    contract_version: str
    case_id: str
    image_path: str
    text: str
    patient_id: Optional[str]
    flags: dict                      # contracts.RouterFlags
    image_result: dict               # contracts.ImageResult
    text_result: dict                # contracts.TextResult
    patient: Optional[dict]          # contracts.PatientRecord
    knowledge: dict                  # contracts.KnowledgeResult
    fused: Optional[dict]            # contracts.FusedResult
    decision: str                    # "ok" | "abstain"
    reasons: list                    # list[contracts.Reason]
    report_md: str
    citations: list                  # list[str] PMIDs
    trace: Annotated[list, operator.add]   # list[contracts.TraceEvent]; parallel nodes append
