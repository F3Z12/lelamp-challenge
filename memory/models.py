"""
Memory Data Models
===================
Simple dataclasses representing what the lamp remembers.
Kept separate from storage logic for clean architecture.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Detection:
    """A single object detected in one frame."""
    label: str
    confidence: float
    x: float            # Normalized center x (0.0 = left, 1.0 = right)
    y: float            # Normalized center y (0.0 = top, 1.0 = bottom)
    width: float        # Normalized bounding box width
    height: float       # Normalized bounding box height

    @property
    def spatial_description(self) -> str:
        """Human-readable position: 'left', 'center', 'right' etc."""
        if self.x < 0.33:
            h = "left"
        elif self.x > 0.66:
            h = "right"
        else:
            h = "center"

        if self.y < 0.33:
            v = "top"
        elif self.y > 0.66:
            v = "bottom"
        else:
            v = "middle"

        return f"{v}-{h}" if v != "middle" else h


@dataclass
class MemoryEntry:
    """A stored observation in the lamp's memory."""
    id: int | None
    label: str
    confidence: float
    spatial_position: str       # e.g. "center", "top-left"
    x: float
    y: float
    first_seen: str             # ISO timestamp
    last_seen: str              # ISO timestamp
    times_seen: int             # How many detection sweeps included this

    def __str__(self):
        return (f"[{self.label}] at {self.spatial_position} "
                f"(seen {self.times_seen}x, last: {self.last_seen})")
