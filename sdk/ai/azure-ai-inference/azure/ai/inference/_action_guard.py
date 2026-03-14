from enum import Enum
from typing import Any


class GuardDecision(Enum):
    ALLOW = "allow"
    BLOCK = "block"


ToolCall = Any
