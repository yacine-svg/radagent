"""knowledge-server (MCP SDK v2, stdio). Run from the repo root:  python -m servers.knowledge_server

Tools are typed with the contract models, so the server publishes the contract's JSON schema
and every response is validated before it leaves the server.
"""
import logging

from mcp.server.mcpserver import MCPServer

from contracts import GuidelineSnippet, ICD10, Reference
from servers import knowledge

log = logging.getLogger("knowledge-server")
mcp = MCPServer("knowledge-server",
                instructions="PubMed references, ICD-10 codes and guideline notes for chest X-ray findings.")


def _valid_list(model, items: list[dict]) -> list:
    out = []
    for item in items:
        try:
            out.append(model.model_validate(item))
        except Exception as e:                            # noqa: BLE001 — drop the bad item, never raise
            log.warning("dropped invalid %s: %s", model.__name__, e)
    return out


@mcp.tool()
def search_pubmed(query: str, max_results: int = 3) -> list[Reference]:
    """Search PubMed. Returns up to max_results references {pmid, title, journal, year}; cached for offline use."""
    return _valid_list(Reference, knowledge.search_pubmed(query, max_results))


@mcp.tool()
def icd10_lookup(condition: str) -> ICD10:
    """ICD-10 code {condition, code, label} for a supported finding; code and label are null when unknown."""
    try:
        return ICD10.model_validate(knowledge.icd10_lookup(condition))
    except Exception as e:                                # noqa: BLE001
        log.warning("invalid ICD-10 row for %r: %s", condition, e)
        return ICD10(condition=str(condition))


@mcp.tool()
def guideline_snippets(condition: str, k: int = 2) -> list[GuidelineSnippet]:
    """Top-k guideline paragraphs {text, source} about a finding (BM25 retrieval over local notes)."""
    return _valid_list(GuidelineSnippet, knowledge.guideline_snippets(condition, k))


if __name__ == "__main__":
    mcp.run(transport="stdio")
