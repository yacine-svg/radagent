"""Helpers shared by all nodes: trace events, the never-raise wrapper, artifact loading."""
from __future__ import annotations

import functools
import inspect
import json
import pathlib
import time

from ml.labels import CLASSES

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "ml" / "artifacts"

DEFAULT_BETA = 1.0              # used until A writes ml/artifacts/beta.json (A5)
DEFAULT_THRESHOLD = 0.5         # used until A writes ml/artifacts/thresholds.json (A5)


def event(node: str, message: str, t0: float, tool: str | None = None) -> dict:
    """One contracts.TraceEvent."""
    return {"node": node, "message": message,
            "duration_ms": max(0, int((time.perf_counter() - t0) * 1000)), "tool": tool}


def root_cause(e: BaseException) -> BaseException:
    """The MCP client wraps errors in ExceptionGroups (task groups): show the real one."""
    while isinstance(e, BaseExceptionGroup) and e.exceptions:
        e = e.exceptions[0]
    return e


def safe_node(name: str, fallback):
    """AGENTS.md 6.1: a node never raises. On an exception, `fallback(state, error_message)`
    returns the node's normal keys with an `error` set, and a trace event records the failure.
    Works for plain and async nodes."""
    def failed(state, e, t0):
        e = root_cause(e)
        msg = f"{type(e).__name__}: {e}"
        out = fallback(state, msg)
        out["trace"] = [event(name, f"FAILED — {msg}", t0)]
        return out

    def wrap(fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def ainner(state):
                t0 = time.perf_counter()
                try:
                    return await fn(state)
                except Exception as e:                   # noqa: BLE001 — by design
                    return failed(state, e, t0)
            return ainner

        @functools.wraps(fn)
        def inner(state):
            t0 = time.perf_counter()
            try:
                return fn(state)
            except Exception as e:                       # noqa: BLE001 — by design
                return failed(state, e, t0)
        return inner
    return wrap


def load_beta() -> float:
    f = ARTIFACTS / "beta.json"
    return float(json.loads(f.read_text())["beta"]) if f.exists() else DEFAULT_BETA


def load_fused_thresholds() -> dict[str, float]:
    f = ARTIFACTS / "thresholds.json"
    if f.exists():
        return {c: float(v) for c, v in json.loads(f.read_text())["fused"].items()}
    return {c: DEFAULT_THRESHOLD for c in CLASSES}


def empty_text_result(error: str | None = None) -> dict:
    """contracts.TextResult for 'no text' (or a failed analysis when error is set)."""
    return {"analyzed_text": "", "entities": [], "suspected": {c: "absent" for c in CLASSES},
            "history_used": False, "error": error}
