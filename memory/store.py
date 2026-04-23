"""
Memory Store
=============
SQLite-backed storage for the lamp's observations.

Key design decisions:
- Deduplication: if the same object label is detected in a similar position
  (within 15% of frame width/height), we update the existing row rather than
  creating a new one. This keeps the DB clean for demo purposes.
- Spatial descriptions are stored as human-readable strings ("top-left",
  "center", etc.) so the LLM can use them directly.
"""

import sqlite3
from datetime import datetime, timezone
from memory.models import Detection, MemoryEntry


class MemoryStore:
    """Persistent memory for the lamp's scene observations."""

    # If a re-detection is within this normalized distance, it's the same object
    DEDUP_DISTANCE = 0.15

    def __init__(self, db_path: str = "memory.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    # ── Public API ──────────────────────────────────────────

    def store_detections(self, detections: list[Detection]) -> int:
        """
        Store a batch of detections, deduplicating against recent entries.
        Returns number of new entries created (vs. updated).
        """
        new_count = 0
        now = datetime.now(timezone.utc).isoformat()

        for det in detections:
            existing = self._find_similar(det.label, det.x, det.y)
            if existing:
                # Update existing entry
                self.conn.execute(
                    """UPDATE memories
                       SET last_seen = ?, times_seen = times_seen + 1,
                           confidence = MAX(confidence, ?), x = ?, y = ?
                       WHERE id = ?""",
                    (now, det.confidence, det.x, det.y, existing.id)
                )
            else:
                # Insert new entry
                self.conn.execute(
                    """INSERT INTO memories
                       (label, confidence, spatial_position, x, y,
                        first_seen, last_seen, times_seen)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                    (det.label, det.confidence, det.spatial_description,
                     det.x, det.y, now, now)
                )
                new_count += 1

        self.conn.commit()
        return new_count

    def get_all(self) -> list[MemoryEntry]:
        """Get all memories, most recently seen first."""
        rows = self.conn.execute(
            "SELECT * FROM memories ORDER BY last_seen DESC"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_by_label(self, label: str) -> list[MemoryEntry]:
        """Search memories by object label (case-insensitive partial match)."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE LOWER(label) LIKE ? ORDER BY last_seen DESC",
            (f"%{label.lower()}%",)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_recent(self, limit: int = 20) -> list[MemoryEntry]:
        """Get the N most recently seen memories."""
        rows = self.conn.execute(
            "SELECT * FROM memories ORDER BY last_seen DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_context_for_llm(self, limit: int = 30) -> str:
        """
        Format stored memories as a text block for the LLM prompt.
        Designed to be concise and directly usable by the model.
        """
        entries = self.get_recent(limit)
        if not entries:
            return "No objects have been observed yet."

        lines = ["Objects the lamp has observed:"]
        for e in entries:
            lines.append(
                f"- {e.label} at {e.spatial_position} of the scene "
                f"(confidence: {e.confidence:.0%}, "
                f"seen {e.times_seen}x, last seen: {e.last_seen})"
            )
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Return summary stats for the HUD."""
        row = self.conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT label) FROM memories"
        ).fetchone()
        return {"total_entries": row[0], "unique_labels": row[1]}

    def close(self):
        self.conn.close()

    # ── Private helpers ─────────────────────────────────────

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                spatial_position TEXT NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                times_seen INTEGER NOT NULL DEFAULT 1
            )
        """)
        self.conn.commit()

    def _find_similar(self, label: str, x: float, y: float) -> MemoryEntry | None:
        """Find an existing entry with the same label near the same position."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE LOWER(label) = LOWER(?)", (label,)
        ).fetchall()
        for r in rows:
            entry = self._row_to_entry(r)
            dist = ((entry.x - x) ** 2 + (entry.y - y) ** 2) ** 0.5
            if dist < self.DEDUP_DISTANCE:
                return entry
        return None

    @staticmethod
    def _row_to_entry(row) -> MemoryEntry:
        return MemoryEntry(
            id=row[0], label=row[1], confidence=row[2],
            spatial_position=row[3], x=row[4], y=row[5],
            first_seen=row[6], last_seen=row[7], times_seen=row[8],
        )
