"""Abstain node + explainer node.

B1: the explainer is a deterministic TEMPLATE built only from the evidence in the state.
B6 adds the LLM explainer on top; this template stays as its fallback when the LLM fails,
so the app always returns a valid, grounded report.
"""
from __future__ import annotations

import time

from contracts import DISCLAIMER, REPORT_SECTIONS
from agents.common import event

ENTITY_TYPES_FR = {"symptom": "symptôme", "history": "antécédent", "finding": "signe"}


def abstain(state: dict) -> dict:
    t0 = time.perf_counter()
    return {"trace": [event("abstain", "Insufficient or contradictory evidence — referred for clinician review", t0)]}


def _first_ref(by_class: dict, label: str):
    refs = by_class.get(label, {}).get("refs", [])
    return refs[0] if refs else None


def build_template_report(state: dict) -> tuple[str, list[str]]:
    fused = state.get("fused")
    reasons = state.get("reasons", [])
    by_class = state["knowledge"].get("by_class", {})
    image = state["image_result"]
    text = state["text_result"]
    heatmaps = image.get("heatmaps", {})

    positives = [f for f in (fused or {}).get("findings", []) if f["above_threshold"]]
    conflict_labels = [r["label"] for r in reasons if r["code"] in ("conflict", "uncertain_conflict")]

    # references: first reference of every positive or conflicting class
    cited, refs_lines = [], []
    for label in [f["label"] for f in positives] + conflict_labels:
        ref = _first_ref(by_class, label)
        if ref and ref["pmid"] not in cited:
            cited.append(ref["pmid"])
            year = f", {ref['year']}" if ref.get("year") else ""
            refs_lines.append(f"- PMID: {ref['pmid']} — {ref['title']} ({ref.get('journal', '')}{year}) [{label}]")

    # Résumé
    if state["decision"] == "abstain":
        blocking = "; ".join(r["message"] for r in reasons if r["severity"] == "abstain")
        summary = f"Le système ne conclut pas : {blocking} Relecture par un radiologue recommandée."
    elif positives:
        summary = "Constat(s) probable(s) : " + ", ".join(f"{f['label']} (probabilité combinée {f['p_fused']:.2f})" for f in positives) + "."
    else:
        summary = "Aucun constat au-dessus des seuils de décision."

    # Observations image
    if image.get("error") is not None:
        obs = f"Analyse d'image indisponible : {image['error']}"
    else:
        lines = []
        for f in fused["findings"]:
            mark = "au-dessus du seuil" if f["above_threshold"] else "sous le seuil"
            cam = " — carte Grad-CAM disponible" if f["label"] in heatmaps else ""
            lines.append(f"- {f['label']} : image {f['p_image']:.2f} → combiné {f['p_fused']:.2f} "
                         f"(seuil {f['threshold']:.2f}, {mark}){cam}")
        obs = "\n".join(lines)

    # Éléments cliniques
    if text.get("error") is not None:
        clinical = f"Analyse du texte indisponible ({text['error']}) ; seules les données image sont utilisées."
    elif not text["analyzed_text"]:
        clinical = "Aucun texte clinique fourni."
    else:
        ents = [f"- {e['text']} ({ENTITY_TYPES_FR[e['type']]}{', nié' if e['negated'] else ''})" for e in text["entities"]]
        clinical = (f"Texte analysé : « {text['analyzed_text']} »\n" + ("\n".join(ents) if ents else "- aucune entité extraite")
                    + ("\nL'historique du patient a été ajouté au texte." if text["history_used"] else ""))

    # Points de vigilance
    vig = [f"- {r['message']}" for r in reasons if r["severity"] in ("abstain", "warning")]
    vig += [f"- Note : {r['message']}" for r in reasons if r["severity"] == "info"]
    icd = [f"{by_class[f['label']]['icd10']['code']} ({f['label']})" for f in positives
           if by_class.get(f["label"], {}).get("icd10", {}).get("code")]
    if icd:
        vig.append("- Codes ICD-10 indicatifs : " + ", ".join(icd))
    vigilance = "\n".join(vig) or "- Aucun point de vigilance."

    body = {
        "## Résumé": summary,
        "## Observations image": obs,
        "## Éléments cliniques": clinical,
        "## Points de vigilance": vigilance,
        "## Références": "\n".join(refs_lines) or "- (aucune)",
        "## Avertissement": DISCLAIMER,
    }
    report = "\n\n".join(f"{h}\n{body[h]}" for h in REPORT_SECTIONS) + "\n"
    return report, cited


def _minimal_report(error: str) -> str:
    body = {h: "- (indisponible)" for h in REPORT_SECTIONS}
    body["## Résumé"] = f"Le rapport n'a pas pu être généré ({error}). Relecture par un radiologue recommandée."
    body["## Avertissement"] = DISCLAIMER
    return "\n\n".join(f"{h}\n{body[h]}" for h in REPORT_SECTIONS) + "\n"


def explainer(state: dict) -> dict:
    t0 = time.perf_counter()
    try:
        report, cited = build_template_report(state)
        msg = f"Report written (template) — {len(cited)} reference(s) cited"
    except Exception as e:                                   # noqa: BLE001 — AGENTS.md 6.1
        report, cited = _minimal_report(f"{type(e).__name__}: {e}"), []
        msg = f"FAILED — minimal report returned ({type(e).__name__})"
    return {"report_md": report, "citations": cited, "trace": [event("explainer", msg, t0)]}
