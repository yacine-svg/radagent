"""vision-server (MCP SDK v2, stdio). Wraps Person A's ml/vision.py. Run:  python -m servers.vision_server

The model is imported on the first call (then stays loaded for the life of the server).
Failures (unreadable image, missing model…) are returned as MCP tool errors (ToolError), which
the image agent turns into image_result.error (contract §4): the server itself never crashes.
"""
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from contracts import VisionPrediction, heatmap_path

mcp = MCPServer("vision-server", instructions="Chest X-ray findings (DenseNet-121) and Grad-CAM heatmaps.")


def _vision():
    from ml import vision                      # Person A's module; loads DenseNet once
    return vision


@mcp.tool()
def classify_image(image_path: str) -> VisionPrediction:
    """Calibrated probabilities for the supported chest X-ray findings."""
    try:
        return VisionPrediction.model_validate(_vision().predict(image_path))
    except Exception as e:                     # noqa: BLE001
        raise ToolError(f"classify_image failed: {type(e).__name__}: {e}") from e


@mcp.tool()
def gradcam(image_path: str, label: str, case_id: str) -> str:
    """Write the Grad-CAM overlay for `label` and return its path (outputs/<case_id>/cam_<label>.png)."""
    try:
        return _vision().gradcam(image_path, label, heatmap_path(case_id, label))
    except Exception as e:                     # noqa: BLE001
        raise ToolError(f"gradcam failed: {type(e).__name__}: {e}") from e


if __name__ == "__main__":
    mcp.run(transport="stdio")
