"""Configuration for KV cache snapshot storage."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _parse_csv_prefixes(raw: str) -> tuple[str, ...]:
    return tuple(p for p in (s.strip() for s in raw.split(",")) if p)


@dataclass
class SnapshotConfig:
    """Configuration for KV cache snapshot storage."""
    enabled: bool = False
    snapshot_dir: str = "/var/cache/vllm/kv"
    warm_max_bytes: int = 8 * 1024 * 1024 * 1024  # 8 GB
    ttl_seconds: int = 86400  # 24 hours

    # Allowed snapshot_id prefixes. When non-empty, every /kv/* operation
    # requires a snapshot_id starting with one of these strings, and
    # POST /kv/snapshot must supply an explicit snapshot_id (the server
    # will not auto-generate one). Empty tuple disables enforcement —
    # callers can use any id, matching pre-hardening behavior.
    allowed_id_prefixes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> SnapshotConfig:
        """Build a config from the VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES env var (CSV)."""
        raw = os.getenv("VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES", "")
        return cls(allowed_id_prefixes=_parse_csv_prefixes(raw))
