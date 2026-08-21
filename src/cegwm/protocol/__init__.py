"""Protocol-owned types that do not depend on method implementations."""

from cegwm.protocol.records import StageARecord, UnitStatus
from cegwm.protocol.stage_a import StageAProtocol, StageAUnit, load_stage_a_protocol

__all__ = [
    "StageAProtocol",
    "StageARecord",
    "StageAUnit",
    "UnitStatus",
    "load_stage_a_protocol",
]
