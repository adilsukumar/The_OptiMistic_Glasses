"""Desktop, browser, YouTube, typing, and application automation modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    """Standard result returned by all automation controllers."""

    success: bool
    message: str = ""
    data: dict[str, Any] | None = None
