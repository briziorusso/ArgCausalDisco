"""Keep corrected graph scores separate from historical evaluations."""
from __future__ import annotations

import json
from pathlib import Path


GRAPH_METRIC_PROTOCOL = "graph_v2"


def require_metric_protocol(record, *, source):
    found = record.get("graph_metric_protocol")
    if found != GRAPH_METRIC_PROTOCOL:
        raise ValueError(
            f"Metric protocol mismatch in {source}: {found!r}; expected "
            f"{GRAPH_METRIC_PROTOCOL!r}. Use a new results version/output path. "
            "Historical scores must be recomputed from saved graphs, not relabelled."
        )


def ensure_metric_protocol(marker, *, existing_paths=()):
    """Allow fresh/current output locations; refuse unlabelled existing results."""
    marker = Path(marker)
    if marker.exists():
        require_metric_protocol(json.loads(marker.read_text(encoding="utf-8")), source=marker)
        return
    existing = [Path(path) for path in existing_paths if Path(path).exists()]
    if existing:
        require_metric_protocol({}, source=existing[0])
    marker.parent.mkdir(parents=True, exist_ok=True)
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"graph_metric_protocol": GRAPH_METRIC_PROTOCOL}, stream, indent=2)
        stream.write("\n")
