from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from datetime import UTC, datetime


_WHITESPACE_RE = re.compile(r"\s+")


def utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def make_prefixed_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    collapsed = _WHITESPACE_RE.sub(" ", ascii_value.strip().lower())
    return collapsed


def stable_embedding(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dimensions:
        for byte in digest:
            values.append((byte / 255.0) * 2 - 1)
            if len(values) == dimensions:
                break
        digest = hashlib.sha256(digest).digest()
    return values


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0

    dot = sum(lv * rv for lv, rv in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def fit_embedding_dimensions(values: list[float], target_dimensions: int) -> list[float]:
    if target_dimensions <= 0:
        return values
    if len(values) == target_dimensions:
        return values
    if len(values) < target_dimensions:
        return [*values, *([0.0] * (target_dimensions - len(values)))]

    reduced: list[float] = []
    step = len(values) / target_dimensions
    for index in range(target_dimensions):
        start = int(index * step)
        end = int((index + 1) * step)
        if end <= start:
            end = start + 1
        chunk = values[start:end]
        reduced.append(sum(chunk) / len(chunk))
    return reduced


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
