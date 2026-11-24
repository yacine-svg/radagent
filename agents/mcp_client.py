"""Small helpers to talk to our MCP servers (MCP SDK v2 `Client`).

    from mcp import Client
    async with Client(stdio("knowledge")) as c:
        icd = tool_output(await c.call_tool("icd10_lookup", {"condition": "Nodule"}))
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any

from mcp import StdioServerParameters

from agents.common import ROOT

SERVER_MODULES = {"vision": "servers.vision_server",
                  "patient": "servers.patient_server",
                  "knowledge": "servers.knowledge_server"}


class ToolCallError(RuntimeError):
    """The server answered with is_error=True."""


def stdio(name: str) -> StdioServerParameters:
    """Launch one of our servers as a subprocess, with the repo root as working directory."""
    return StdioServerParameters(command=sys.executable, args=["-m", SERVER_MODULES[name]], cwd=str(ROOT))


def target(name: str):
    """What to connect to: a subprocess over stdio (the real app), or the server object itself
    when RADAGENT_MCP=inprocess (tests: same code path, no subprocess, much faster)."""
    if os.environ.get("RADAGENT_MCP") == "inprocess":
        return importlib.import_module(SERVER_MODULES[name]).mcp
    return stdio(name)


def tool_output(result) -> Any:
    """CallToolResult -> plain Python (dict or list), whatever way the server encoded it."""
    text = "".join(getattr(c, "text", "") for c in (result.content or []))
    if result.is_error:
        raise ToolCallError(text or "tool error")
    sc = result.structured_content
    if sc is not None:
        return sc["result"] if isinstance(sc, dict) and set(sc) == {"result"} else sc
    return json.loads(text)
