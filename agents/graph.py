"""RadAgent graph (LangGraph).

    router ─┬─ image_agent ─────┐
            ├─ text_agent ──────┼─ fusion ─ critic ─┬──────────── explainer
            └─ knowledge_agent ─┘                   └─ abstain ──┘

B1 status: image/text/knowledge agents are FAKES (agents/fake_agents.py, data from examples/).
Router, fusion, critic, abstain and the template explainer are real.

Run:  python -m agents.graph                 # the 3 example cases, validated against the contract
      python -m agents.graph --case abstain  # one case
      python -m agents.graph --sequential    # without LangGraph (same node order)
"""
from __future__ import annotations

import argparse
import json
import sys

from agents.router import router
from agents.fake_agents import image_agent, text_agent, knowledge_agent   # B3: swap for the real agents
from agents.fusion import fusion
from agents.critic import critic
from agents.explainer import abstain, explainer

PARALLEL = [("image_agent", image_agent), ("text_agent", text_agent), ("knowledge_agent", knowledge_agent)]


def build_graph():
    """Compile the LangGraph app (imported lazily so the tests run without LangGraph)."""
    from langgraph.graph import END, START, StateGraph
    from agents.state import CaseState

    g = StateGraph(CaseState)
    g.add_node("router", router)
    for name, fn in PARALLEL:
        g.add_node(name, fn)
    g.add_node("fusion", fusion)
    g.add_node("critic", critic)
    g.add_node("abstain", abstain)
    g.add_node("explainer", explainer)

    g.add_edge(START, "router")
    for name, _ in PARALLEL:
        g.add_edge("router", name)          # fan-out: the 3 agents run in parallel
    g.add_edge([name for name, _ in PARALLEL], "fusion")   # fan-in: fusion waits for all 3
    g.add_edge("fusion", "critic")
    g.add_conditional_edges("critic", lambda s: s["decision"], {"ok": "explainer", "abstain": "abstain"})
    g.add_edge("abstain", "explainer")
    g.add_edge("explainer", END)
    return g.compile()


def _merge(state: dict, update: dict) -> dict:
    out = {**state, **{k: v for k, v in update.items() if k != "trace"}}
    out["trace"] = state.get("trace", []) + update.get("trace", [])
    return out


def run_sequential(inputs: dict) -> dict:
    """Same nodes, same order, no LangGraph. Used by CI and as a debugging fallback."""
    state = _merge({"trace": []}, router(inputs))
    after_router = dict(state)
    for _, fn in PARALLEL:                  # each agent sees the state right after the router
        state = _merge(state, fn(after_router))
    state = _merge(state, fusion(state))
    state = _merge(state, critic(state))
    if state["decision"] == "abstain":
        state = _merge(state, abstain(state))
    return _merge(state, explainer(state))


_APP = None


def run(inputs: dict, sequential: bool = False) -> dict:
    global _APP
    if sequential:
        return run_sequential(inputs)
    if _APP is None:
        _APP = build_graph()
    return _APP.invoke(inputs)


# ------------------------------------------------------------------ CLI
CASES = {"agreement": "case_agreement", "conflict": "case_conflict_warning", "abstain": "case_abstain"}


def example_inputs(case: str) -> dict:
    from agents.common import ROOT
    ex = json.loads((ROOT / "examples" / f"{CASES[case]}.json").read_text(encoding="utf-8"))
    return {k: ex[k] for k in ("case_id", "image_path", "text", "patient_id")}


def main(argv=None) -> int:
    from contracts import FinalState
    ap = argparse.ArgumentParser(description="Run RadAgent on the example cases.")
    ap.add_argument("--case", choices=[*CASES, "all"], default="all")
    ap.add_argument("--sequential", action="store_true", help="run without LangGraph")
    ap.add_argument("--show-report", action="store_true")
    args = ap.parse_args(argv)

    ok = True
    for case in (CASES if args.case == "all" else [args.case]):
        final = run(example_inputs(case), sequential=args.sequential)
        try:
            FinalState.model_validate(final)
            status = "valid ✓"
        except Exception as e:                               # show every contract problem
            status, ok = f"INVALID ✗\n{e}", False
        print(f"\n[{case}] decision={final.get('decision')}  contract: {status}")
        for t in final.get("trace", []):
            print(f"   {t['node']:<16} {t['message']}")
        if args.show_report:
            print("\n" + final.get("report_md", ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
