"""RadAgent — Streamlit app.  Run from the repo root:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                    # all contract paths are relative to the repo root

import streamlit as st                            # noqa: E402

try:
    from dotenv import load_dotenv                # GROQ_API_KEY / GROQ_MODEL for the LLM explainer
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import agents.graph as G                          # noqa: E402
from app.render import (cited_refs, decision_banner, finding_label, highlight_html,  # noqa: E402
                        report_for_display, trace_line)
from contracts import CONTRACT_VERSION, DISCLAIMER, FinalState  # noqa: E402
from servers.patients import list_patient_ids    # noqa: E402

HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None
DEMOS = {"— none —": None, "Agreement (cardiomegaly + CHF history)": "agreement",
         "Conflict (image: effusion, text: no effusion)": "conflict",
         "Abstention (uncertain conflict)": "abstain"}

st.set_page_config(page_title="RadAgent", page_icon="🩻", layout="wide")
st.title("🩻 RadAgent")
st.caption("Agentic decision support for chest X-ray analysis — image model + clinical text + MCP evidence")
st.info(f"**{DISCLAIMER}**")


# ------------------------------------------------------------------ sidebar: inputs
def load_demo():
    case = DEMOS[st.session_state.demo]
    if case:
        inp = G.example_inputs(case)
        st.session_state.text = inp["text"]
        st.session_state.patient = inp["patient_id"] or "—"


with st.sidebar:
    st.header("Case")
    st.selectbox("Demo case", list(DEMOS), key="demo", on_change=load_demo)
    upload = st.file_uploader("Chest X-ray (PNG/JPG)", type=["png", "jpg", "jpeg"])
    st.text_area("Clinical text (indication / history)", key="text", height=110,
                 placeholder="e.g. 62-year-old woman, shortness of breath, history of CHF")
    st.selectbox("Patient (synthetic, optional)", ["—", *list_patient_ids()], key="patient")
    run_clicked = st.button("Analyse", type="primary")

    with st.expander("Settings"):
        transport = st.radio("MCP transport", ["In-process (fast)", "stdio subprocesses"],
                             help="Same MCP servers and protocol. In-process keeps the image model loaded between cases.")
        explainer_mode = st.radio("Report writer", ["LLM (Groq), template fallback", "Template only"],
                                  disabled=not os.environ.get("GROQ_API_KEY"),
                                  help="Needs GROQ_API_KEY in .env")
        st.caption("Agents: " + ", ".join(
            f"{n.replace('_agent', '')} = {'fake' if fn.__module__.endswith('fake_agents') else 'real'}"
            for n, fn in G.PARALLEL))
        st.caption(f"Orchestration: {'LangGraph' if HAS_LANGGRAPH else 'sequential runner (LangGraph not installed)'}")


# ------------------------------------------------------------------ run
def image_for_run() -> str | None:
    case = DEMOS[st.session_state.demo]
    if upload is not None:
        path = Path("outputs/uploads") / f"{uuid.uuid4().hex[:8]}{Path(upload.name).suffix.lower()}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(upload.getvalue())
        return path.as_posix()
    if case:
        return G.example_inputs(case)["image_path"]
    return None


if run_clicked:
    image_path = image_for_run()
    if image_path is None:
        st.warning("Upload an X-ray or choose a demo case first.")
        st.stop()
    case = DEMOS[st.session_state.demo]
    inputs = {"case_id": G.example_inputs(case)["case_id"] if case and upload is None else None,
              "image_path": image_path, "text": st.session_state.text,
              "patient_id": None if st.session_state.patient == "—" else st.session_state.patient}
    os.environ["RADAGENT_MCP"] = "inprocess" if transport.startswith("In-process") else ""
    os.environ["RADAGENT_EXPLAINER"] = "template" if explainer_mode.startswith("Template") else "llm"

    with st.status("Agents at work…", expanded=True) as status:
        def on_update(node, update):
            for ev in update.get("trace", []):
                st.markdown(trace_line(ev))
        try:
            final = asyncio.run(G.arun(inputs, sequential=not HAS_LANGGRAPH, on_update=on_update))
            status.update(label=f"Done — decision: {final['decision']}", state="complete", expanded=False)
        except Exception as e:                     # noqa: BLE001 — show it, don't crash the app
            status.update(label="Run failed", state="error")
            st.exception(e)
            st.stop()
    st.session_state.final = final


# ------------------------------------------------------------------ results
final = st.session_state.get("final")
if not final:
    st.markdown("Choose a **demo case** or upload an X-ray, add the clinical text, then click **Analyse**.")
    st.stop()

kind, message = decision_banner(final)
getattr(st, kind)(f"**{message}**")
for r in final["reasons"]:
    {"abstain": st.error, "warning": st.warning, "info": st.caption}[r["severity"]](f"{r['code']}: {r['message']}")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Image")
    img = final["image_result"]
    heatmaps = {k: v for k, v in img.get("heatmaps", {}).items() if Path(v).exists()}
    shown = final["image_path"]
    if heatmaps and st.toggle("Show Grad-CAM", value=True):
        label = st.selectbox("Heatmap for", list(heatmaps)) if len(heatmaps) > 1 else next(iter(heatmaps))
        shown = heatmaps[label]
        st.caption(f"Grad-CAM: regions that raised the **{label}** score. A clue, not a proof.")
    if Path(shown).exists():
        st.image(shown)
    else:
        st.info(f"Image file not available here: `{shown}`")
    if img.get("error"):
        st.error(f"Image analysis failed: {img['error']}")

with right:
    st.subheader("Findings")
    if final.get("fused"):
        for f in final["fused"]["findings"]:
            st.progress(f["p_fused"], text=finding_label(f))
        st.caption(f"Combined = image evidence shifted by the clinical text (β = {final['fused']['beta']}). "
                   "✔ above the decision threshold · ⚠ image and text disagree")
    else:
        st.info("No image result, so nothing to combine.")

    st.subheader("Clinical text")
    tr = final["text_result"]
    if tr.get("error"):
        st.warning(f"Text analysis failed: {tr['error']}")
    elif tr["analyzed_text"]:
        st.markdown(highlight_html(tr["analyzed_text"], tr["entities"]), unsafe_allow_html=True)
        st.caption("🟨 symptom · 🟦 history · 🟪 finding · ~~struck~~ = negated"
                   + (" · patient history added" if tr["history_used"] else ""))
    else:
        st.caption("No clinical text given.")
    if final.get("patient") and final["patient"].get("found"):
        p = final["patient"]
        st.caption(f"Patient {p['patient_id']} (synthetic): {p['age']} y, {p['sex']}, "
                   f"{'smoker' if p['smoker'] else 'non-smoker'} — prior: {'; '.join(p['prior_studies']) or 'none'}")

st.divider()
st.subheader("Report")
st.markdown(report_for_display(final["report_md"]))
st.download_button("Download report (.md)", final["report_md"], file_name=f"{final['case_id']}_report.md",
                   mime="text/markdown")

with st.expander(f"References cited ({len(final['citations'])})"):
    refs = cited_refs(final)
    for r in refs:
        st.markdown(f"- [PMID {r['pmid']}]({r['url']}) — {r['title']} *({r['journal']}, {r['year']})* · {r['about']}")
    if not refs:
        st.caption("No references cited (offline and PubMed cache empty? run `python -m servers.warm_cache`).")

with st.expander("Agent trace"):
    for ev in final["trace"]:
        st.markdown(trace_line(ev))

with st.expander("Raw output (data contract)"):
    try:
        FinalState.model_validate(final)
        st.success(f"Output matches data contract v{CONTRACT_VERSION}")
    except Exception as e:                          # noqa: BLE001
        st.error(f"Contract violation: {e}")
    st.json(json.loads(json.dumps(final, default=str)), expanded=False)

st.caption(DISCLAIMER)
