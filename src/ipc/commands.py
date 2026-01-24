import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShutdownCommand:
    graceful: bool = True
    timeout: float = 5.0


@dataclass(frozen=True)
class HealthCheckCommand:
    pass


@dataclass(frozen=True)
class ReindexDocumentCommand:
    doc_id: str
    reason: str = ""


@dataclass(frozen=True)
class IndexUpdatedNotification:
    version: int
    doc_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class InitCompleteNotification:
    version: int
    doc_count: int
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class HealthStatusResponse:
    healthy: bool
    queue_depth: int
    last_index_time: float | None
    doc_count: int


IPCMessage = (
    ShutdownCommand
    | HealthCheckCommand
    | ReindexDocumentCommand
    | IndexUpdatedNotification
    | InitCompleteNotification
    | HealthStatusResponse
)
