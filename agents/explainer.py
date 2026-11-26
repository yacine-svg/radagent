"""Abstain node + explainer node.

The explainer writes the French report in two possible ways:
  1. LLM (Groq), when GROQ_API_KEY is set: it only sees the evidence JSON, and its output goes
     through 3 grounding guards (fixed headings, only known PMIDs, exact disclaimer);
  2. TEMPLATE: deterministic, built only from the evidence. Used when there is no key,
     when RADAGENT_EXPLAINER=template, or as the fallback whenever the LLM fails a guard.
Either way the app always returns a valid, grounded report.
"""
from __future__ import annotations

import json
import os
import re
import time

from contracts import DISCLAIMER, PMID_PATTERN, REPORT_SECTIONS
from agents.common import event
from servers.knowledge import french_name

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


# ------------------------------------------------------------------ LLM explainer
LLM_SYSTEM = f"""Tu rédiges le compte rendu d'un assistant d'AIDE à l'analyse de radiographies thoraciques.
Tu reçois UNIQUEMENT un JSON de preuves produit par d'autres agents. Règles strictes :
1. Écris en français, en Markdown, avec EXACTEMENT ces 6 titres, dans cet ordre, sans autre titre :
{chr(10).join(REPORT_SECTIONS)}
2. N'utilise que les informations du JSON. N'invente aucun constat, chiffre, antécédent ou référence.
3. Nomme chaque constat avec son nom français exact donné dans "names_fr" (ex. « épanchement pleural », jamais « pleurésie »).
4. Si "decision" vaut "ok" : le Résumé présente les constats avec "above_threshold": true comme « probables », avec leur probabilité combinée, ou dit qu'aucun constat ne dépasse les seuils. N'écris PAS que le système « ne conclut pas ».
   Si "decision" vaut "abstain" : le Résumé dit clairement que le système ne conclut pas, et pourquoi (d'après "reasons").
5. Chaque élément de "reasons" (conflits, avertissements) apparaît dans « Points de vigilance », reformulé en français.
6. Dans « Références », une ligne par référence fournie dans "references" : « - PMID: 12345678 — titre (journal, année) ». Aucun autre PMID.
7. La section « ## Avertissement » contient exactement cette phrase : {DISCLAIMER}
8. Ton prudent : jamais de diagnostic définitif, pas de conseil de traitement."""


def evidence(state: dict) -> dict:
    """The only thing the LLM sees: the evidence relevant to this case, nothing else."""
    fused = state.get("fused")
    reasons = state.get("reasons", [])
    by_class = state["knowledge"].get("by_class", {})
    relevant = [f["label"] for f in (fused or {}).get("findings", []) if f["above_threshold"]]
    relevant += [r["label"] for r in reasons if r.get("label") and r["label"] not in relevant]
    text = state["text_result"]
    return {
        "decision": state["decision"],
        "reasons": [{k: r[k] for k in ("code", "severity", "label", "message")} for r in reasons],
        "findings": [{k: f[k] for k in ("label", "p_image", "s_text", "p_fused", "threshold", "above_threshold")}
                     | {"conflict": f["conflict"]["message"] if f["conflict"] else None}
                     for f in (fused or {}).get("findings", [])],
        "image_error": state["image_result"].get("error"),
        "heatmaps_for": sorted(state["image_result"].get("heatmaps", {})),
        "clinical_text": {"analyzed_text": text["analyzed_text"], "history_used": text["history_used"],
                          "error": text.get("error"),
                          "entities": [{k: e[k] for k in ("text", "type", "negated")} for e in text["entities"]]},
        "icd10": {lab: by_class[lab]["icd10"] for lab in relevant if lab in by_class},
        "guidelines": {lab: by_class[lab]["guideline"] for lab in relevant
                       if lab in by_class and by_class[lab].get("guideline")},
        "references": [r | {"about": lab} for lab in relevant if lab in by_class for r in by_class[lab]["refs"]],
        "names_fr": {f["label"]: french_name(f["label"]) for f in (fused or {}).get("findings", [])},
    }


class GroundingError(ValueError):
    """The LLM output broke a grounding rule: the template is used instead."""


def ground(report: str, allowed_pmids: set[str]) -> tuple[str, list[str]]:
    """Guards applied to every LLM report. Returns (report, citations) or raises GroundingError."""
    report = report.strip()
    report = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", report).strip()       # stray code fences
    report = re.sub(r"PMID\s*[:#]?\s*(\d{1,9})", r"PMID: \1", report)          # one citation format
    pos = [report.find(h) for h in REPORT_SECTIONS]
    if -1 in pos or pos != sorted(pos):                                          # guard 1: structure
        raise GroundingError("missing or misordered headings")
    unknown = set(PMID_PATTERN.findall(report)) - allowed_pmids                  # guard 2: references
    if unknown:
        report = "\n".join(l for l in report.splitlines()
                           if not (set(PMID_PATTERN.findall(l)) & unknown))
    head, _, _ = report.partition("## Avertissement")                             # guard 3: disclaimer
    report = f"{head.rstrip()}\n\n## Avertissement\n{DISCLAIMER}\n"
    cited = list(dict.fromkeys(PMID_PATTERN.findall(report)))
    return report, cited


def make_llm():
    """Groq client, or None when the LLM is not configured (tests replace this function)."""
    if os.environ.get("RADAGENT_EXPLAINER") == "template" or not os.environ.get("GROQ_API_KEY"):
        return None
    from groq import Groq
    return Groq(timeout=30)


def llm_report(state: dict, client) -> tuple[str, list[str]]:
    ev = evidence(state)
    r = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", ""), temperature=0,
        messages=[{"role": "system", "content": LLM_SYSTEM},
                  {"role": "user", "content": json.dumps(ev, ensure_ascii=False)}])
    return ground(r.choices[0].message.content or "", {ref["pmid"] for ref in ev["references"]})


def explainer(state: dict) -> dict:
    t0 = time.perf_counter()
    how = "template"
    try:
        client = make_llm()
        if client is not None:
            try:
                report, cited = llm_report(state, client)
                how = "LLM"
            except Exception as e:                       # noqa: BLE001 — fall back to the template
                how = f"template (LLM fallback: {type(e).__name__})"
                report, cited = build_template_report(state)
        else:
            report, cited = build_template_report(state)
        msg = f"Report written ({how}) — {len(cited)} reference(s) cited"
    except Exception as e:                               # noqa: BLE001 — AGENTS.md 6.1
        report, cited = _minimal_report(f"{type(e).__name__}: {e}"), []
        msg = f"FAILED — minimal report returned ({type(e).__name__})"
    return {"report_md": report, "citations": cited, "trace": [event("explainer", msg, t0, "groq" if how == "LLM" else None)]}
