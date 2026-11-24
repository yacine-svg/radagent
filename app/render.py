"""Pure helpers for the Streamlit app (no Streamlit import, so they are unit-tested)."""
from __future__ import annotations

import html

ENTITY_COLORS = {"symptom": "#FEF3C7", "history": "#E0E7FF", "finding": "#FCE7F3"}
NODE_ICONS = {"router": "🧭", "image_agent": "🩻", "text_agent": "📝", "knowledge_agent": "📚",
              "fusion": "🔀", "critic": "🛡️", "abstain": "✋", "explainer": "🧾"}


def highlight_html(text: str, entities: list[dict]) -> str:
    """analyzed_text with every entity span highlighted by type; negated entities struck through."""
    out, pos = [], 0
    for e in sorted(entities, key=lambda e: e["span"][0]):
        s, t = e["span"]
        if s < pos:                                       # overlapping span: skip it
            continue
        out.append(html.escape(text[pos:s]))
        frag = html.escape(text[s:t])
        if e["negated"]:
            frag = f"<s>{frag}</s>"
        title = f"{e['type']}{' (negated)' if e['negated'] else ''}"
        out.append(f'<mark title="{title}" style="background:{ENTITY_COLORS.get(e["type"], "#EEE")};'
                   f'padding:0 3px;border-radius:4px">{frag}</mark>')
        pos = t
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def finding_label(f: dict) -> str:
    """One line per fused finding for the progress bars."""
    text = {1: "text +", -1: "text −", 0: "text ·"}[f["s_text"]]
    marks = (" ✔" if f["above_threshold"] else "") + (" ⚠ conflict" if f["conflict"] else "")
    return (f"**{f['label']}** — combined {f['p_fused']:.2f} (image {f['p_image']:.2f}, {text}; "
            f"threshold {f['threshold']:.2f}){marks}")


def trace_line(event: dict) -> str:
    tool = f" · `{event['tool']}`" if event.get("tool") else ""
    return f"{NODE_ICONS.get(event['node'], '•')} **{event['node']}** — {event['message']} ({event['duration_ms']} ms){tool}"


def pubmed_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def cited_refs(final: dict) -> list[dict]:
    """The references actually cited in the report, with the class they support."""
    out = []
    for label, k in final["knowledge"].get("by_class", {}).items():
        for r in k["refs"]:
            if r["pmid"] in final.get("citations", []) and r["pmid"] not in {x["pmid"] for x in out}:
                out.append(r | {"about": label, "url": pubmed_url(r["pmid"])})
    return out


def decision_banner(final: dict) -> tuple[str, str]:
    """(kind, message) — kind is 'success' or 'error' (Streamlit alert type)."""
    if final["decision"] == "abstain":
        return "error", "No conclusion: insufficient or contradictory evidence — referred for clinician review."
    positives = [f["label"] for f in (final.get("fused") or {}).get("findings", []) if f["above_threshold"]]
    return "success", ("Probable finding(s): " + ", ".join(positives)) if positives else "No finding above the decision thresholds."
