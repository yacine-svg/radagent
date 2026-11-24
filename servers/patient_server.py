"""patient-server (MCP SDK v2, stdio). SYNTHETIC records only. Run:  python -m servers.patient_server"""
import logging

from mcp.server.mcpserver import MCPServer

from contracts import PatientRecord
from servers import patients

log = logging.getLogger("patient-server")
mcp = MCPServer("patient-server", instructions="Synthetic patient histories for the RadAgent demo.")


@mcp.tool()
def get_patient_history(patient_id: str) -> PatientRecord:
    """Patient record {patient_id, found, age, sex, smoker, history, prior_studies}; found=false if unknown."""
    try:
        return PatientRecord.model_validate(patients.get_patient_history(patient_id))
    except Exception as e:                                # noqa: BLE001
        log.warning("invalid patient record for %r: %s", patient_id, e)
        return PatientRecord(patient_id=str(patient_id), found=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
