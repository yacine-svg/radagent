"""RadAgent graph (LangGraph).

    router ─┬─ image_agent ─────┐
            ├─ text_agent ──────┼─ fusion ─ critic ─┬──────────── explainer
            └─ knowledge_agent ─┘                   └─ abstain ──┘

Agents (B3): knowledge and text agents are real (MCP servers). The image agent is real as soon as
ml/vision.py exists (A2); until then it is the fake one. RADAGENT_FAKE=1 forces all three fakes.

Run:  python -m agents.graph                 # the 3 example cases, validated against the contract
      python -m agents.graph --case abstain  # one case
      python -m agents.graph --sequential    # without LangGraph (same node order)
      python -m agents.graph --fake          # fake agents only (no MCP servers)
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import os
import sys

from agents import fake_agents
from agents.router import router
from agents.fusion import fusion
from agents.critic import critic
from agents.explainer import abstain, explainer



def select_agents(fake: bool | None = None) -> list:
    """[(name, node)] for the 3 parallel agents: real where possible, fake otherwise."""
    if fake is None:
        fake = os.environ.get("RADAGENT_FAKE") == "1"
    if fake:
        return [("image_agent", fake_agents.image_agent), ("text_agent", fake_agents.text_agent),
                ("knowledge_agent", fake_agents.knowledge_agent)]
    from agents import real_agents
    has_vision = importlib.util.find_spec("ml.vision") is not None
    return [("image_agent", real_agents.image_agent if has_vision else fake_agents.image_agent),
            ("text_agent", real_agents.text_agent), ("knowledge_agent", real_agents.knowledge_agent)]


PARALLEL = select_agents()


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


async def _call(fn, state):
    out = fn(state)
    return await out if inspect.isawaitable(out) else out


async def arun_sequential(inputs: dict) -> dict:
    """Same nodes, same order, no LangGraph. Used by CI and as a debugging fallback."""
    state = _merge({"trace": []}, await _call(router, inputs))
    after_router = dict(state)              # each agent sees the state right after the router
    for update in await asyncio.gather(*(_call(fn, after_router) for _, fn in PARALLEL)):
        state = _merge(state, update)
    state = _merge(state, await _call(fusion, state))
    state = _merge(state, await _call(critic, state))
    if state["decision"] == "abstain":
        state = _merge(state, await _call(abstain, state))
    return _merge(state, await _call(explainer, state))


async def arun(inputs: dict, sequential: bool = False) -> dict:
    """Run one case (async). The app and the CLI use this."""
    if sequential:
        return await arun_sequential(inputs)
    return await build_graph().ainvoke(inputs)


def run(inputs: dict, sequential: bool = False) -> dict:
    """Blocking wrapper around arun()."""
    return asyncio.run(arun(inputs, sequential))


def run_sequential(inputs: dict) -> dict:
    return asyncio.run(arun_sequential(inputs))


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
    ap.add_argument("--fake", action="store_true", help="fake agents only (no MCP servers)")
    ap.add_argument("--show-report", action="store_true")
    args = ap.parse_args(argv)
    global PARALLEL
    PARALLEL = select_agents(fake=True if args.fake else None)
    print("agents:", ", ".join(f"{n}={'fake' if fn.__module__.endswith('fake_agents') else 'real'}" for n, fn in PARALLEL))

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
