from __future__ import annotations

from collections import Counter
from typing import Any


def corpus_statistics(recordings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "recording_count": len(recordings),
        "duration_hours": sum(int(x.get("duration_samples", 0)) for x in recordings) / 48000 / 3600,
        "by_status": dict(Counter(x.get("status", "unknown") for x in recordings)),
        "by_split": dict(Counter(x.get("split", "unset") for x in recordings)),
        "by_room": dict(Counter(x.get("room_id") or "unset" for x in recordings)),
        "by_environment": dict(Counter(x.get("environment_id") or "unset" for x in recordings)),
    }


def assign_grouped_splits(items: list[dict[str, Any]], *, tolerance: float = 0.05) -> dict[str, str]:
    if not items:
        return {}
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    seen: dict[tuple[str, str], int] = {}
    for i, item in enumerate(items):
        values = [("session", item.get("capture_session_id")), ("room", item.get("room_id"))] + [
            ("speaker", x) for x in item.get("speaker_ids_anonymous", [])
        ]
        for kind, value in values:
            if value:
                key = (kind, str(value))
                union(i, seen[key]) if key in seen else seen.setdefault(key, i)
    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)
    total = sum(max(1, int(x.get("duration_samples", 1))) for x in items)
    targets = {"train": 0.70 * total, "validation": 0.15 * total, "test": 0.15 * total}
    used = {k: 0 for k in targets}
    result = {}
    ordered = sorted(groups.values(), key=lambda g: -sum(max(1, int(items[i].get("duration_samples", 1))) for i in g))
    for group in ordered:
        split = min(targets, key=lambda key: used[key] / targets[key])
        size = sum(max(1, int(items[i].get("duration_samples", 1))) for i in group)
        used[split] += size
        for i in group:
            result[items[i]["id"]] = split
    ratios = {k: used[k] / total for k in used}
    if len(items) >= 20 and any(abs(ratios[k] - targets[k] / total) > tolerance for k in ratios):
        raise ValueError(f"分组后split比例偏差超过5个百分点: {ratios}")
    return result
